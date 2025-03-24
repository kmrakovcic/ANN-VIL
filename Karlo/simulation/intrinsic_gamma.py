import numpy as np
import pickle
import scipy
import scipy.stats
from scipy.stats.sampling import NumericalInversePolynomial
from scipy import constants
from scipy import integrate
from scipy.integrate import quad


class intrinsicLC:
    def __init__(self, a):
        self.a = a

    def support(self):
        # distribution restricted to 0, 5, can be changed as needed
        return (0, 7200)

    def pdf(self, x):
        # this is not a proper pdf, the normalizing
        # constant is missing (does not integrate to one)
        return np.exp(-0.5 * ((x - self.a[1]) / self.a[2]) ** 2) * self.a[0] + np.exp(
            -0.5 * ((x - self.a[4]) / self.a[5]) ** 2) * self.a[3]


def powerLaw(a, b, x0, alpha, size=1):
    g = 1 - alpha
    #"""Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a ** g, b ** g
    #return (ag + (bg - ag)*(r/x0))**(1./g)
    return (ag + (bg - ag) * r) ** (1. / g)


def intrinsic_times(A1, mean1, sigma1, A2, mean2, sigma2, size=10 ** 6):
    sigma1 = max(200, sigma1)
    sigma2 = max(200, sigma2)
    if mean2 < mean1:
        A_tmp, sigma_tmp, mean_tmp = A1, sigma1, mean1
        A1, sigma1, mean1 = A2, sigma2, mean2
        A2, sigma2, mean2 = A_tmp, sigma_tmp, mean_tmp
    distLC = intrinsicLC([A1, mean1, sigma1, A2, mean2, sigma2])
    genLC = NumericalInversePolynomial(distLC)
    const_pdf = quad(distLC.pdf, *distLC.support())[0]
    return genLC.rvs(size=size)


def intrinsic_energy(a, b, E0, alpha, size=10 ** 6):
    return powerLaw(a, b, E0, alpha, size)
