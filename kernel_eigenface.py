
"""
"""

from common import *
import numpy as np
import math

import pdb

IMAGES = ["data/{idx}.png".format(idx=str(i + 1)) for i in range(20)]
DEBUG = True


def polynomial_kernel(img1, img2, p=2):

    return (1 + sum(img1 * img2))**p


def gaussian_kernel(img1, img2, beta=1):

    return math.pow(
        math.e,
        -1 * beta * np.linalg.norm(np.subtract(img2, img1))**2)


class KPCA:

    def __init__(self, data, kernel):

        self.data = data
        self.kernel_matrix = self._make_kernel_matrix(data, kernel)
        self.kernel = kernel

    def run(self):

        (self.eigenvalues,
         self.eigenvectors) = np.linalg.eigh(self.kernel_matrix)

        self.eigenvectors = self.eigenvectors / len(self.data)

        return self

    def _make_kernel_matrix(self, data, kernel):

        K = np.matrix([[kernel(i, j) for i in data] for j in data])
        n = K.shape[0]
        J = np.matrix(np.ones((n, n)) * (1 / n))

        return K - 2 * J * K + J.T * K * J

    def project(self, x):

        y = [
            float(
                np.dot(alpha, [self.kernel(x[0], x_i) for x_i in self.data]))
            for alpha in self.eigenvectors
        ]

        return np.matrix(y).T


if __name__ == "__main__":

    # Load images
    data = load_images(IMAGES) / 256 / 50 / 50

    kpca = KPCA(data, polynomial_kernel).run()

    # for i in range(10):
    #    A = np.matrix(eigenvectors[i]).T
    #    p = data.T * A
    #    show_face(p.reshape([50, 50]), caption="Eigenvector {i}".format(i=i))

    for file in ['data/1.png', 'testcase/test_01.png', 'testcase/test_02.png']:

        src = normalize(load_gray(file).reshape([1, -1]))

        projected = kpca.project(src)
