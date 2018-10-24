
"""
"""

import eigenface
import numpy as np
import math


def polynomial_kernel(img1, img2, p=2):

    return (1 + sum(np.multiply(img1, img2)))**p


def gaussian_kernel(img1, img2, beta=1):

    return math.pow(
        math.e,
        -1 * beta * np.linalg.norm(np.subtract(img2, img1))**2)


def make_kernel_matrix(data, kernel):

    return [[kernel(i, j) for i in data] for j in data]
