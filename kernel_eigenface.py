
"""
"""

from common import *
import numpy as np
import math

IMAGES = ["data/{idx}.png".format(idx=str(i + 1)) for i in range(640)]
DEBUG = True


def polynomial_kernel(img1, img2, p=2):

    return (1 + sum(img1 * img2))**p


def gaussian_kernel(img1, img2, beta=1):

    return math.pow(
        math.e,
        -1 * beta * np.linalg.norm(np.subtract(img2, img1))**2)


def make_kernel_matrix(data, kernel):

    K = np.matrix([[kernel(i, j) for i in data] for j in data])
    n = K.shape[0]
    J = np.matrix(np.ones((n, n)) * (1 / n))

    return K - 2 * J * K + J.T * K * J


def project(image, data, eigenvalues, eigenvectors, kernel):

    # A <- eigenvectors (each eigenvector is a row)
    A = np.matrix(eigenvectors)
    # K <- [K(x, x_i)...].T
    K = np.matrix([kernel(image[0], s) for s in data]).T

    # y_j  = sum(a_ji K(x, x_i))
    #      = a_j.T * K
    # => y = A * K
    weights = A * K

    return normalize(np.matrix(data).T * (A.T * weights))


if __name__ == "__main__":

    # Load images
    data = load_images(IMAGES)

    K = make_kernel_matrix(data, polynomial_kernel)
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    for file in ['testcase/test_01.png', 'testcase/test_02.png']:

        src = normalize(load_gray(file).reshape([1, -1]))

        p = project(
            src, data, eigenvalues[:2], eigenvectors[:20], polynomial_kernel)

        error = np.absolute(p.T - src)

        show_face(p.reshape([50, 50]), caption="Approximated Image")
        show_face(
            error.reshape([50, 50]),
            caption="Error: {err} ({perr}%)"
            .format(err=np.sum(error), perr=np.sum(error) / 2500))
