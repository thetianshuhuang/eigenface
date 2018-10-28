
import cv2
import numpy as np
import kernels
import math


class PCA:

    def __init__(self, data):

        self.data = data

    def run(self):

        (self.mean,
         self.eigenvectors,
         self.eigenvalues) = cv2.PCACompute2(self.data, mean=None)

        return self

    def project(self, x, n):

        projection = [np.sum(np.multiply(x, v)) for v in self.eigenvectors[:n]]
        approx = sum(
            np.multiply(*v)
            for v in zip(projection, self.eigenvectors))

        return approx


class KPCA:
    r"""Kernel PCA object

    Parameters
    ----------
    data : np.array
        Data matrix, where each data point is a row
    kernel : function ($R^N x R^N -> R$)
        Kernel function to use; should take two vectors as input

    Attributes
    ----------
    data : np.array
        Data matrix
    kernel_matrix : np.array
        Computed kernel matrix
    kernel : np.array
        Kernel used
    eigenvectors : np.array
        Array of eigenvectors; each eigenvector is a row.
        Sorted by increasing order of eigenvalue.
    eigenvalues : np.array
        Array of eigenvalues
    """

    def __init__(self, data, kernel):

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

        Returns
        -------
        self
            A reference to this object. Use for method chaining.
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

        Returns
        -------
        float [beta_k]
            Projection of the input vector along a principle component in the
            kernel space.
        """

        return np.dot(
            alpha_k, np.array([self.kernel.get(x, x_i) for x_i in self.data]))

    def _gamma_i(self, x, i, n):
        r"""Compute the intermediate term gamma_i.

        \[\gamma_i = \sum_{k=1}^N \beta_k \alpha_{ki}\]

        Parameters
        ----------
        x : np.array
            New vector to project
        i : int
            Index of the gamma value to fetch

        Returns
        -------
        float [gamma_i]
            Gamma_i constant used in iterative estimator
        """

        assert 0 <= i and i < self.N

        return np.dot(
            np.array([
                self._beta_k(x, alpha_k)
                for alpha_k in self.eigenvectors[-n:]]),
            np.array([
                alpha_k[i] for alpha_k in self.eigenvectors[-n:]])
        )

    def project(self, x, n, random_start=False, iterations=20):
        r"""Compute the pre-image of a data vector in the input space

        Parameters
        ----------
        x : np.array
            Input array
        n : int
            Number of eigenvectors to use

        Keyword Args
        ------------
        random_start : bool
            If True, a random vector is used for the initial vector. Otherwise,
            the input image is used in order to speed up convergence.
        iterations : int
            Number of iterations to run the iterative estimator.

        Returns
        -------
        np.array
            Estimated projection of x to the eigenvector subspace in the kernel
            space, and back to the input space

        References
        ----------
        "Kernel PCA and De-Noising in Feature Spaces", S. Mika et Al, 1999
        """

        assert self.computed
        assert len(x.shape) == 1 and x.shape[0] == self.d

        z_k = (
            np.array([np.random.random() for i in range(self.d)])
            if random_start else x)
        gammas = [self._gamma_i(x, i, n) for i in range(self.N)]

        for i in range(iterations):
            z_k = self.kernel.iter(gammas, self.data, z_k)

        return z_k
