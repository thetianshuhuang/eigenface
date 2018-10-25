
"""
Compute eigenfaces, and test reference images by projecting to the top N
eigenfaces.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from common import *


IMAGES = ["data/{idx}.png".format(idx=str(i + 1)) for i in range(640)]
PCA_RANK = 20
PCA_SCAN_INTERVAL = 5
DEBUG = True


def project(image, eigenvectors):
    """Project a face onto a set of eigenvectors.

    Parameters
    ----------
    image : array
        image to project
    eigenvectors : np.array[]
        list of eigenvectors.
    """

    projection = [np.sum(np.multiply(image, v)) for v in eigenvectors]
    approx = sum(np.multiply(*v) for v in zip(projection, eigenvectors))

    show_face(
        approx.reshape([50, 50]),
        caption="Approximated Image")

    error = np.absolute(np.subtract(image, approx))
    show_face(
        error.reshape([50, 50]),
        caption="Error: {err} ({perr}%)"
        .format(err=np.sum(error), perr=np.sum(error) / 2500 / 256))


def test(file, eigenvectors):
    """Run tests on a given file using a set of eigenvectors

    Parameters
    ----------
    file : str
        File to load test image from
    eigenvectors : np.array[]
        Complete list of all eigenvectors
    """

    for i in range(1, int(PCA_RANK / PCA_SCAN_INTERVAL)):
        print(i * PCA_SCAN_INTERVAL)
        project(
            load_gray(file).reshape([1, -1]),
            eigenvectors[:PCA_SCAN_INTERVAL * i])


if __name__ == "__main__":

    # Load images
    data = load_images(IMAGES)

    # Show data matrix
    if DEBUG:
        cv2.imshow('Data Matrix', data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Run PCA
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data, mean=None)

    # Show top 5 eigenfaces
    if DEBUG:
        print(eigenvectors[0])
        for i in range(5):
            show_face(
                eigenvectors[i].reshape([50, 50]),
                caption="Eigenvector {idx}: Value={val}"
                .format(idx=str(i + 1), val=eigenvalues[i]))

    # Run tests
    test("testcase/test_01.png", eigenvectors)
    test("testcase/test_02.png", eigenvectors)

    # Show Eigenspectrum
    plt.plot([i for i in range(640)], eigenvalues)
    plt.show()

    plt.plot([i for i in range(640)], [math.log(abs(i)) for i in eigenvalues])
    plt.show()
