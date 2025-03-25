import sys
sys.path.append('../')
from simulation import dataset_creation
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate opacity grid")
    parser.add_argument('--E0_min', type=float,
                        default=11,
                        help='Lower bound for first parameter of spectrum parametric form. Write only exponent of 10 potential')
    parser.add_argument('--E0_max', type=float,
                        default=13,
                        help='Upper bound for first parameter of spectrum parametric form. Write only exponent of 10 potential')
    parser.add_argument('--alpha_min', type=float,
                        default=1,
                        help='Lower bound for second parameter of spectrum parametric form.')
    parser.add_argument('--alpha_max', type=float,
                        default=3,
                        help='Upper bound for second parameter of spectrum parametric form.')

    parser.add_argument('--A1_min', type=float,
                        default=10,
                        help='Lower bound for first parameter of light curve parametric form.')
    parser.add_argument('--A1_max', type=float,
                        default=100,
                        help='Upper bound for first parameter of light curve parametric form.')
    parser.add_argument('--mean1_min', type=float,
                        default=1000,
                        help='Lower bound for second parameter of light curve parametric form.')
    parser.add_argument('--mean1_max', type=float,
                        default=5000,
                        help='Upper bound for second parameter of light curve parametric form.')
    parser.add_argument('--sigma1_min', type=float,
                        default=600,
                        help='Lower bound for third parameter of light curve parametric form.')
    parser.add_argument('--sigma1_max', type=float,
                        default=5000,
                        help='Upper bound for third parameter of light curve parametric form.')

    parser.add_argument('--A2_min', type=float,
                        default=10,
                        help='Lower bound for fourth parameter of light curve parametric form.')
    parser.add_argument('--A2_max', type=float,
                        default=100,
                        help='Upper bound for fourth parameter of light curve parametric form.')
    parser.add_argument('--mean2_min', type=float,
                        default=1000,
                        help='Lower bound for fifth parameter of light curve parametric form.')
    parser.add_argument('--mean2_max', type=float,
                        default=5000,
                        help='Upper bound for fifth parameter of light curve parametric form.')
    parser.add_argument('--sigma2_min', type=float,
                        default=600,
                        help='Lower bound for sixth parameter of light curve parametric form.')
    parser.add_argument('--sigma2_max', type=float,
                        default=5000,
                        help='Upper bound for sixth parameter of light curve parametric form.')


    parser.add_argument('--opacity_file', type=str,
                        default="../extra/Opacity_grid_100x100.npz",
                        help='File with opacity grid for interpolation')

    parser.add_argument('--photon_num', type=int,
                        default=2000,
                        help='Number of photons in each simulation')
    parser.add_argument('--examples_num', type=int,
                        default=10000,
                        help='Number of examples in data set')
    parser.add_argument('--verbose', type=argparse.BooleanOptionalAction,
                        default=True,
                        help='Print verbose.')
    parser.add_argument('--output_suffix', type=str,
                        default="",
                        help='Add to the end of the output name.')
    args = parser.parse_args()

    x, y = dataset_creation.create_train_set(args.examples_num,
                            [(10 ** args.E0_min, args.alpha_min), (10 ** args.E0_max, args.alpha_max)],
                            [(args.A1_min, args.mean1_min, args.sigma1_min, args.A2_min, args.mean2_min, args.sigma2_min),
                             (args.A1_max, args.mean1_max, args.sigma1_max, args.A2_max, args.mean2_max, args.sigma2_max)],
                            interpolation_grid_file=args.opacity_file,
                            parallel=True,
                            photon_num=args.photon_num,
                            verbose=args.verbose)
    examples_num = x.shape[0]
    np.savez("../extra/trainset_p"+str(args.photon_num)+"n"+str(examples_num)+"_"+args.output_suffix+".npz", x=x, y=y)

if __name__ == '__main__':
    main()