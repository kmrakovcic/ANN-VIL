import os
import numpy as np
import pickle
import random
import scipy
import scipy.stats
import datetime
from scipy.stats.sampling import NumericalInversePolynomial
from scipy import constants
from scipy import integrate
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit

Mpc2km = 10 ** 3 * constants.parsec
H0 = 70. / Mpc2km
Om_m = 0.3
Om_L = 0.7


def distanceContrib(zs=0.034, n=2):
    dist = lambda z: np.power(1 + z, n) / np.sqrt(Om_L + Om_m * np.power(1 + z, 3))
    return integrate.quad(dist, 0, zs)[0]


def timeDelay(E, L, kappa2):
    if L == 0:
        delta = 0.
    else:
        delta = 1.5 / H0 * np.power(E / L, 2) * kappa2
    return delta


def detectionEnergy(E, z):
    return E / (1. + z)
