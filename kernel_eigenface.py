
"""Sample file for pca.KPCA"""

import sys
import pca
import faces
import kernels
from common import timeit, show_error


@timeit
def kpca_model(kernel):
    return pca.KPCA(faces.IMAGES, kernel).run()


@timeit
def project(src, model):
    return model.project(src, 20)


if __name__ == "__main__":

    k = (
        kernels.Polynomial() if (
            len(sys.argv) >= 1 and "polynomial" in sys.argv)
        else kernels.Gaussian())
    model = kpca_model(k)

    # Run tests
    for i, image in enumerate(faces.TEST_IMAGES):
        proj = project(image, model)
        show_error(image, proj, [50, 50], "test_" + str(i))
