
"""Sample file for pca.KPCA"""

import sys
import pca
import mnist
import kernels
from common import timeit, show_error


@timeit
def kpca_model(kernel):
    return pca.KPCA(mnist.IMAGES[:1000], kernel).run()


@timeit
def project(src, model):
    return model.project(src, 20)


if __name__ == "__main__":

    # Generate model with polynomial or gaussian kernel
    k = (
        kernels.Polynomial() if (
            len(sys.argv) >= 1 and "polynomial" in sys.argv)
        else kernels.Gaussian())
    model = kpca_model(k)

    # Run tests
    for i, image in enumerate(mnist.IMAGES[-10:]):
        proj = project(image, model)
        show_error(image, proj, [28, 28], "test_" + str(i))
