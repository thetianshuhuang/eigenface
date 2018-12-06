
"""Sample file for pca.PCA"""

import pca
import faces
from common import timeit, show_error, show_face


@timeit
def pca_model():
    return pca.PCA(faces.IMAGES).run()


@timeit
def project(src, model):
    return model.project(src, 20)


if __name__ == "__main__":

    model = pca_model()

    # Show top 5 eigenvectors
    for i, eigenface in enumerate(model.eigenvectors[:5]):
        show_face(
            eigenface.reshape([50, 50]),
            caption="Eigenvector #{n}".format(n=i + 1))

    # Run tests
    for image in faces.TEST_IMAGES:
        proj = project(image, model)
        show_error(image, proj, [50, 50])
