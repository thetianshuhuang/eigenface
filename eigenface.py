
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


IMAGES = ["data/{idx}.png".format(idx=str(i + 1)) for i in range(640)]
PCA_RANK = 20
PCA_SCAN_INTERVAL = 5
DEBUG = False


def show_face(image, caption="Image"):

    # Copy for display transformations
    disp = image.copy()

    # Scale to [0,1]
    disp = np.multiply(disp, 1 / (disp.max() - disp.min()))
    disp = np.subtract(disp, disp.min())

    # Make bigger and show
    disp = cv2.resize(disp, None, fx=10, fy=10)
    cv2.imshow(caption, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_gray(filepath):

    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY)


def load_images():

    data = None
    for file in IMAGES:
        img = load_gray(file)
        img = img.reshape([1, -1])
        if data is None:
            data = img.copy()
            print(data)
        else:
            data = np.concatenate((data, img), axis=0)

    return data


def project(image, eigenvectors):

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
    for i in range(1, int(PCA_RANK / PCA_SCAN_INTERVAL)):
        print(i * PCA_SCAN_INTERVAL)
        project(
            load_gray(file).reshape([1, -1]),
            eigenvectors[:PCA_SCAN_INTERVAL * i])


data = load_images()

if DEBUG:
    cv2.imshow('Data Matrix', data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

mean, eigenvectors, eigenvalues = cv2.PCACompute2(data, mean=None)

if DEBUG:
    for i in range(5):
        show_face(
            eigenvectors[i].reshape([50, 50]),
            caption="Eigenvector {idx}: Value={val}"
            .format(idx=str(i + 1), val=eigenvalues[i]))

test("testcase/test_01.png", eigenvectors)
test("testcase/test_02.png", eigenvectors)


plt.plot([i for i in range(640)], eigenvalues)
plt.show()

plt.plot([i for i in range(640)], [math.log(abs(i)) for i in eigenvalues])
plt.show()
