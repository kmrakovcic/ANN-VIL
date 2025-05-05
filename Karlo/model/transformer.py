import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import GradScaler, autocast
import numpy as np

# -------- Load and preprocess data --------
def load_from_file (data_path):
    x_train, y_train = np.load(data_path).values()
    return x_train, y_train

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=2000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # Match sequence length

class ResidualGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out * 2)
        self.shortcut = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, x):
        residual = self.shortcut(x)
        x_proj, gate = self.fc(x).chunk(2, dim=-1)
        x = x_proj * torch.sigmoid(gate)
        return self.bn(x + residual)

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.repeat(batch_size, 1, 1)  # [batch_size, 1, embed_dim]
        attn_output, attn_weights = self.mha(query, x, x)
        return attn_output.squeeze(1)  # [batch_size, embed_dim]


class LIVTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_layers=4, ff_dim=128, max_len=2000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = MultiHeadAttentionPooling(embed_dim, num_heads=num_heads)

        self.head = nn.Sequential(
            ResidualGLU(embed_dim, 128),
            nn.GELU(),
            ResidualGLU(128, 64),
            nn.Softsign(),
            ResidualGLU(64, 32),
            nn.Softsign(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 2),
        )

        # Normalization parameters
        self.register_buffer('x_max_time', torch.tensor(1.0))
        self.register_buffer('x_mean_energies', torch.tensor(0.0))
        self.register_buffer('x_std_energies', torch.tensor(1.0))
        self.register_buffer('y_min', torch.tensor(0.0))
        self.register_buffer('y_max', torch.tensor(1.0))

    def calibrate_normalization(self, x, y):
        # Normalize x
        times = x[:, :, 0]
        energies = x[:, :, 1] * 1e-10
        self.x_max_time = torch.tensor(times.max())
        self.x_mean_energies = torch.tensor(energies.mean())
        self.x_std_energies = torch.tensor(energies.std())

        # Normalize y
        y = np.log10(y)
        self.y_min = torch.tensor(y.min())
        self.y_max = torch.tensor(y.max())

    def normalize_x_internal(self, x):
        times = x[:, :, 0] / self.x_max_time
        energies = (x[:, :, 1] * 1e-10 - self.x_mean_energies) / self.x_std_energies
        return torch.stack((times, energies), dim=-1)

    def normalize_y_internal(self, y):
        y = torch.log10(y)
        return (y - self.y_min) / (self.y_max - self.y_min)

    def unnormalize_y_internal(self, y):
        return y * (self.y_max - self.y_min) + self.y_min

    def forward(self, x):
        x = self.normalize_x_internal(x)
        x = self.embedding(x) + self.positional_encoding(x)
        x = self.encoder(x)
        x = self.attn_pool(x)
        out = self.head(x)
        mean = out[:, 0]
        var = torch.abs(out[:, 1])
        return torch.stack((mean, var), dim=1)

    def train_model(self, x_train, y_train, save_path, epochs=200, batch_size=256, lr=1e-4, patience=20, train_split=0.75):
        self.calibrate_normalization(x_train, y_train)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_train_tensor = self.normalize_y_internal(y_train_tensor)
        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.apply(init_weights)

        mse_criterion = nn.MSELoss()
        criterion = marginal_posterior_density_loss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)
        scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

        best_loss = np.inf
        trigger_times = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    pred = self(xb)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * xb.size(0)

            self.eval()
            val_loss = 0.0
            val_mse_loss = 0.0
            y_pred_list, y_true_list = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = self(xb)
                    y_pred_list.append(pred.cpu().numpy())
                    y_true_list.append(yb.cpu().numpy())
                    loss = criterion(pred, yb)
                    mse_loss = mse_criterion(
                        self.unnormalize_y_internal(pred[:, 0]),
                        self.unnormalize_y_internal(yb)
                    )
                    val_loss += loss.item() * xb.size(0)
                    val_mse_loss += mse_loss.item() * xb.size(0)

            y_pred = np.concatenate(y_pred_list, axis=0)
            y_true = np.concatenate(y_true_list, axis=0)
            cov = coverage(y_true, y_pred)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            val_mse_loss /= len(val_loader.dataset)

            scheduler.step()

            print(
                f"Epoch {epoch + 1:03d} - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, Val MSE: {val_mse_loss:.5f}, Coverage: {cov:.5f}, LR: {scheduler.get_last_lr()[0]:.7f}")

            if epoch > patience:
                if val_loss < best_loss:
                    best_loss = val_loss
                    trigger_times = 0
                    torch.save(self.state_dict(), save_path)
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        self.load_state_dict(torch.load(save_path))

    def finetune_head(self, x_train, y_train, save_path, epochs=200, batch_size=256, lr=1e-4, train_split=0.75):
        self.calibrate_normalization(x_train, y_train)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_train_tensor = self.normalize_y_internal(y_train_tensor)
        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for param in self.encoder.parameters():
            param.requires_grad = False

        criterion = marginal_posterior_density_loss()
        mse_criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler()

        best_loss = np.inf

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    pred = self(xb)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * xb.size(0)

            self.eval()
            val_loss = 0.0
            val_mse_loss = 0.0
            y_pred_list, y_true_list = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = self(xb)
                    y_pred_list.append(pred.cpu().numpy())
                    y_true_list.append(yb.cpu().numpy())
                    loss = criterion(pred, yb)
                    mse_loss = mse_criterion(
                        self.unnormalize_y_internal(pred[:, 0]),
                        self.unnormalize_y_internal(yb)
                    )
                    val_loss += loss.item() * xb.size(0)
                    val_mse_loss += mse_loss.item() * xb.size(0)

            y_pred = np.concatenate(y_pred_list, axis=0)
            y_true = np.concatenate(y_true_list, axis=0)
            cov = coverage(y_true, y_pred)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            val_mse_loss /= len(val_loader.dataset)

            scheduler.step()

            print(
                f"Finetune Epoch {epoch + 1:03d} - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, Val MSE: {val_mse_loss:.5f}, Coverage: {cov:.5f}, LR: {scheduler.get_last_lr()[0]:.7f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.state_dict(), save_path)

        self.load_state_dict(torch.load(save_path))

    def predict(self, x, batch_size=256):
        x = torch.tensor(x, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        y_pred_list = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size].to(device)
                batch_pred = self(batch)
                batch_pred[:, 0] = self.unnormalize_y_internal(batch_pred[:, 0])
                batch_pred[:, 1] = batch_pred[:, 1] * (self.y_max - self.y_min)
                y_pred_list.append(batch_pred.cpu().numpy())

        # Concatenate predictions and apply exponential transformation
        y_pred = np.concatenate(y_pred_list, axis=0)
        return y_pred


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def logarithmic_marginal_posterior_density_loss():
    def loss(y_pred, y_true):
        pred_mean = y_pred[:,0]
        pred_var = y_pred[:,1]

        true_mean = y_true
        moment_loss = torch.log(torch.mean((pred_mean - true_mean) ** 2)) + torch.log(torch.mean(((pred_mean - true_mean) ** 2 - pred_var ** 2)**2))
        return moment_loss
    return loss

def marginal_posterior_density_loss():
    def loss(y_pred, y_true):
        pred_mean = y_pred[:,0]
        pred_var = y_pred[:,1]

        true_mean = y_true
        moment_loss = torch.mean((pred_mean - true_mean) ** 2 + ((pred_mean - true_mean) ** 2 - pred_var ** 2)**2)
        return moment_loss
    return loss


def coverage (y_true, y_pred):
    y_true = torch.tensor(y_true)
    mu = torch.tensor(y_pred[:, :1]) # first output neuron
    sig = torch.abs(torch.tensor(y_pred[:, 1:])) # second output neuron
    return torch.mean((torch.abs(y_true-mu)<torch.abs(sig)).to(torch.float))

if __name__ == '__main__':
    x_train, y_train = np.load("../../Karlo/extra/trainset_p2000n10000_train.npz").values()
    x_test, y_test = np.load("../../Karlo/extra/trainset_p2000n2500_test.npz").values()
    y_train = y_train[:, -1] # only the first output neuron
    y_test = y_test[:, -1] # only the first output neuron
    model = LIVTransformer()
    model.train_model(x_train, y_train, "../../Karlo/extra/transformer55.pt", epochs=200, lr=1e-4, patience=20)
    model.finetune_head(x_train, y_train, "../../Karlo/extra/transformer55.pt", epochs=64, lr=1e-6)