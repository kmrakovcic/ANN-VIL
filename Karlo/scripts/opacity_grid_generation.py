import sys
sys.path.append('../')
from simulation import dataset_creation
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate opacity grid")
    parser.add_argument('--E_min', type=float,
                        default=10.5,
                        help='Lower bound for gamma ray energy. Write only exponent of 10 potential')
    parser.add_argument('--E_max', type=float,
                        default=13.7,
                        help='Upper bound for gamma ray energy. Write only exponent of 10 potential')
    parser.add_argument('--Eqg_min', type=float,
                        default=9,
                        help='Lower bound for LIV scale energy. Write only exponent of 10 potential')
    parser.add_argument('--Eqg_max', type=float,
                        default=30,
                        help='Upper bound for LIV scale energy. Write only exponent of 10 potential')
    parser.add_argument('--E_num', type=int,
                        default=1000,
                        help='Number of gridpoint in E direction')
    parser.add_argument('--Eqg_num', type=int,
                        default=1000,
                        help='Number of gridpoint in Eqg direction')
    parser.add_argument('--verbose', type=argparse.BooleanOptionalAction,
                        default=True,
                        help='Print verbose.')
    args = parser.parse_args()
    opacity_grid, E_grid, Eqg_grid = dataset_creation.create_opacity_grid(args.E_min, args.E_max,
                                                                          args.Eqg_min, args.Eqg_max,
                                                                          N=(args.E_num, args.Eqg_num),
                                                                          verbose=args.verbose)
    np.savez("../extra/Opacity_grid_"+str(args.E_num)+"x"+str(args.Eqg_num)+".npz",
             opacity=opacity_grid, E=E_grid, Eqg=Eqg_grid)

if __name__ == '__main__':
    main()