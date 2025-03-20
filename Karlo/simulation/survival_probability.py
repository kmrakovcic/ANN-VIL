from scipy import constants
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import time
import os

# Necessary constants
if "EBL_PATH" not in os.environ:
    ebl_path = os.getcwd().split("ANN-VIL")[0]+"ANN-VIL/Karlo/extra/EBL_Sal.csv"
else:
    ebl_path = os.environ["EBL_PATH"]
pi = constants.pi
c = constants.speed_of_light
m_e = 10 ** 6 * constants.physical_constants['electron mass energy equivalent in MeV'][0]
me2 = m_e ** 2
hb = constants.physical_constants['Planck constant in eV/Hz'][0] / 2 / pi
kb = constants.Boltzmann
ev2J = constants.electron_volt
alpha = constants.fine_structure
Mp = 10 ** 9 * constants.physical_constants['Planck mass energy equivalent in GeV'][0]
Mpc2km = 10 ** 3 * constants.parsec
H0 = 70 * hb / Mpc2km
Om_m = 0.3
Om_L = 0.7
SEDconvert = 4 * pi / ev2J / 10 ** 9 / c / 5.067 ** 3 / 10 ** 18

def readCSV(fileName):
    return (np.genfromtxt(fileName, delimiter=','))


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
def interpolRGI(EBLen, z, EBLn):
    return RegularGridInterpolator((EBLen, z), EBLn, method='linear', bounds_error=False, fill_value=0.0)


def hub(z):
    return np.sqrt(Om_m * np.power(1 + z, 3) + Om_L)


def muBar(E, L):
    return np.square(np.square(E) / (2 * m_e * L))


def F_LIV(tauB, muB):
    return 4 * pi * np.square(alpha) * ((2 + (2 * tauB * (1 - 2 * muB) - (1 - muB)) / np.square(tauB + muB)) * np.log(
        (1 + np.sqrt(1 - 1 / tauB)) / (1 - np.sqrt(1 - 1 / tauB))) - (
                                                    2 + 2 * tauB * (1 - 4 * muB) / np.square(tauB + muB)) * np.sqrt(
        1 - 1 / tauB))


def integral_LIV(omega, tauB, z, muB):
    return F_LIV(tauB, muB * np.power(1 + z, 4)) * photDens(np.array([omega, z])) / np.square(omega) / np.power(1 + z,
                                                                                                                3) / hub(
        z)


def F_SR(tauB):
    return 4 * pi * np.square(alpha) * ((2 + (2 * tauB - 1) / np.square(tauB)) * np.log(
        (1 + np.sqrt(1 - 1 / tauB)) / (1 - np.sqrt(1 - 1 / tauB))) - (2 + 2 / tauB) * np.sqrt(1 - 1 / tauB))


def integral_SR(omega, tauB, z):
    return F_SR(tauB) * photDens(np.array([omega, z])) / np.square(omega) / np.power(1 + z, 3) / hub(z)


def opacity(E, zs, L):
    if L == 0:
        op = integrate.tplquad(integral_SR, 0, zs, lambda z: 1, lambda z: np.inf,
                               lambda z, tauB: me2 * tauB / (1 + z) / E, lambda z, tauB: np.inf)
    else:
        muB = muBar(E, L)
        op = integrate.tplquad(integral_LIV, 0, zs, lambda z: 1, lambda z: np.inf,
                               lambda z, tauB: me2 * (tauB + muB * np.power(1 + z, 4)) / (1 + z) / E,
                               lambda z, tauB: np.inf, args=(muB,))
    return me2 / 4 / np.square(E) / H0 * op[0]


# Reading EBL table and calculating EBL photon density
redshift = np.array(
    [0., 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
     3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.])
dataEBL = readCSV(ebl_path)
dataEBL = np.flip(dataEBL, 0)
EBLwavelength = dataEBL[:, 0]
EBLen = 2 * pi * hb * c / EBLwavelength * 10 ** 6
EBLSED = dataEBL[:, 1:38] * SEDconvert
EBLn = EBLSED / np.square(EBLen[:, None])
photDens = interpolRGI(EBLen, redshift, EBLn)
