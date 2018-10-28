
import numpy as np
import kernels
import math
import pdb


class KPCA:

    def __init__(self, data, kernel):
        r"""Create a Kernel PCA object.

        Parameters
        ----------
        data : np.array
            Data matrix, where each data point is a row
        kernel : function ($R^N x R^N -> R$)
            Kernel function to use; should take two vectors as input
        """

        self.data = data
        self.kernel_matrix = kernels.matrix(data, kernel)
        self.kernel = kernel

        self.N, self.d = data.shape

        self.computed = False

    def run(self):
        r"""Run Kernel PCA

        PCA is first run on the kernel matrix. Then, the eigenvectors are
        normalized such that
        \[ \lambda_k(\alpha^k \cdot \alpha^k) = 1 \]
        """

        (self.eigenvalues, eigenvectors) = np.linalg.eigh(self.kernel_matrix)

        self.eigenvectors = np.array([
            alpha_k * (1 if lambda_k >= 0 else -1) / math.sqrt(abs(lambda_k))
            for lambda_k, alpha_k in zip(self.eigenvalues, eigenvectors)
        ])
        self.eigenvalues = [abs(lambda_k) for lambda_k in self.eigenvalues]

        self.computed = True

        return self

    def _beta_k(self, x, alpha_k):
        r"""Project a vector onto a principle component.

        \[ \beta_k = \sum_{i=1}^N \alpha_{ki} k(x, x_i) = \alpha_i \cdot k_x \]

        Parameters
        ----------
        x : np.array
            New vector to project
        alpha : np.array
            Eigenvector to project to; alpha_k
        """

        return np.dot(
            alpha_k, np.array([self.kernel(x, x_i) for x_i in self.data]))

    def _gamma_i(self, x, i, n):
        r"""Compute the intermediate term gamma_i.

        \[\gamma_i = \sum_{k=1}^N \beta_k \alpha_{ki}\]

        Parameters
        ----------
        x : np.array
            New vector to project
        i : int
            Index of the gamma value to fetch
        """
        # add n here as well

        assert 0 <= i and i < self.N

        return np.dot(
            np.array([
                self._beta_k(x, alpha_k)
                for alpha_k in self.eigenvectors[-n:]]),
            np.array([
                alpha_k[i] for alpha_k in self.eigenvectors[-n:]])
        )

    def pre_image(self, x, n):
        # add n (number of eigenvectors) here

        assert self.computed
        assert len(x.shape) == 1 and x.shape[0] == self.d

        z_k = np.array([np.random.random() for i in range(self.d)])
        gammas = [self._gamma_i(x, i, n) for i in range(self.N)]

        for i in range(20):
            z_k = self.kernel(gammas, self.data, z_k, iter=True)

        return z_k
