
import numpy as np
import math
import pdb


def _polynomial_k(img1, img2, p=2):

    return (1 + sum(img1 * img2))**p


def gaussian(*args, beta=1, **kwargs):

    if "iter" in kwargs and kwargs["iter"]:
        assert len(args) == 3
        return _gaussian_iter(args[0], args[1], args[2], beta=beta)

    else:
        assert(len(args) == 2)
        return _gaussian_k(args[0], args[1], beta=beta)


def _gaussian_k(img1, img2, beta=1):

    return math.pow(
        math.e,
        -1 * beta * np.linalg.norm(np.subtract(img2, img1))**2)


def _gaussian_iter(gammas, data, z_k, beta=1):
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
    """

    return (
        sum(
            gamma_i * _gaussian_k(z_k, x_i) * x_i
            for gamma_i, x_i in zip(gammas, data)
        ) / sum(
            gamma_i * _gaussian_k(z_k, x_i)
            for gamma_i, x_i in zip(gammas, data))
    )


def matrix(data, kernel):

    K = np.array([[kernel(i, j) for i in data] for j in data])
    n = K.shape[0]
    J = np.array(np.ones((n, n)) * (1 / n))

    return K - 2 * J @ K + J.T @ K @ J
