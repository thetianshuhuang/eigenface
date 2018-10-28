
from kpca import KPCA
from faces import IMAGES, TEST_IMAGES
import kernels
from common import show_face, timeit
import numpy as np


@timeit
def kpca_model():
    return KPCA(IMAGES, kernels.gaussian).run()


@timeit
def project(src, model):
    return model.pre_image(src, 20)


def project_analysis(src, model):
    proj = project(src, model)
    print(src)
    print(proj)
    error = np.absolute(np.subtract(src, proj))
    show_face(src.reshape([50, 50]), caption="Source Image")
    show_face(proj.reshape([50, 50]), caption="Approximated Image")
    show_face(
        error.reshape([50, 50]),
        caption="Error: {err} ({perr})%"
        .format(err=np.sum(error), perr=np.sum(error) / 2500))


model = kpca_model()
for image in TEST_IMAGES:
    project_analysis(image, model)
