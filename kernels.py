
import numpy as np
import math


class Polynomial:

    def __init__(self, p=2):
        self.p = p

    def get(self, img1, img2):
        return (1 + np.dot(img1, img2))**self.p

    def iter(self, gammas, data, z_k):
        r"""
        """
        return sum(
            gamma_i *
            ((np.dot(z_k, x_i) + 1) / (np.dot(z_k, z_k) + 1))**(self.p - 1) *
            x_i
            for gamma_i, x_i in zip(gammas, data)
        )


class Gaussian:

    def __init__(self, beta=1):
        self.beta = 1

    def get(self, img1, img2):
        return math.pow(
            math.e,
            -1 * self.beta * np.linalg.norm(np.subtract(img2, img1))**2)

    def iter(self, gammas, data, z_k):
        r"""Iterative estimator for the gaussian kernel:

        \[
            z_{k + 1} = \frac{
                \sum_{i=1}^N \lambda_i k(z_k, x_i) x_i}
                {\sum_{i=1}^N \lambda_i k(z_k, x_i)}
        \]

        Parameters
        ----------
        gammas : float[]
            gamma_i (computed by KPCA)
        data : np.array
            Data matrix
        z_k : np.array
            Previous estimate
        beta : float
            kernel parameter

        References
        "Kernel PCA and De-Noising in Feature Spaces", S. Mika et Al, 1999
        """

        return (
            sum(
                gamma_i * self.get(z_k, x_i) * x_i
                for gamma_i, x_i in zip(gammas, data)
            ) / sum(
                gamma_i * self.get(z_k, x_i)
                for gamma_i, x_i in zip(gammas, data))
        )


def matrix(data, kernel):

    K = np.array([[kernel.get(i, j) for i in data] for j in data])
    n = K.shape[0]
    J = np.array(np.ones((n, n)) * (1 / n))

    return K - 2 * J @ K + J.T @ K @ J
