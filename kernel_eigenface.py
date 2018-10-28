
"""Sample file for pca.KPCA"""

import pca
import faces
import kernels
from common import timeit, show_error


@timeit
def kpca_model():
    return pca.KPCA(faces.IMAGES, kernels.Polynomial()).run()


@timeit
def project(src, model):
    return model.project(src, 20)


model = kpca_model()

# Run tests
for image in faces.TEST_IMAGES:
    proj = project(image, model)
    show_error(image, proj, [50, 50])
