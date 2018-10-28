
"""
"""

import numpy as np
import time
import cv2


def timeit(f):
    def timed(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)

        print("{f}: {t}s".format(f=f.__name__, t=time.time() - start))
        return res

    return timed


def normalize(image):
    """Normalize an image to [0,1]

    Parameters
    ----------
    image : np.array
        Image to normalize

    Returns
    -------
    """

    image = np.multiply(image, 1 / (image.max() - image.min()))
    image = np.subtract(image, image.min())

    return image


def show_face(image, caption="Image"):
    """Display a face in an OpenCV Window

    Parameters
    ----------
    image : np.array
        Image to display

    Keyword Args
    ------------
    caption : str
        window caption
    """

    # Copy for display transformations
    disp = image.copy()

    disp = normalize(image)

    # Make bigger and show
    disp = cv2.resize(disp, None, fx=10, fy=10)
    cv2.imshow(caption, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_gray(filepath):
    """Load a grayscale image

    Parameters
    ----------
    filepath : str
        File to load

    Returns
    -------
    np.array
        Loaded image
    """

    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY) / 256


def load_images(images):
    """Load all images, as specified by config constant IMAGES

    Returns
    -------
    np.array
        Data matrix, where each row is a data vector consisting of the image
        pixels rearranged in a 2500x1 vector
    """

    data = None
    for file in images:
        img = load_gray(file)
        img = img.reshape([1, -1])
        if data is None:
            data = img.copy()
        else:
            data = np.concatenate((data, img), axis=0)

    return data


def show_error(src, proj, size, save=None):
    """Show approximation error

    Parameters
    ----------
    src : np.array
        Input array
    proj : np.array
        Output array
    size : int[2]
        Width and height of image
    """

    assert len(size) == 2

    error = np.absolute(np.subtract(src, proj))
    scalar_error = np.sum(error) / size[0] / size[1]
    caption = (
        "Error: {err} ({perr}%)"
        .format(err=scalar_error, perr=scalar_error * 100))

    print(caption)

    if save is not None:
        cv2.imwrite(save + ".PNG", proj.reshape(size) * 255)
        cv2.imwrite(save + "_error.PNG", error.reshape(size) * 255)
    else:
        show_face(src.reshape(size), caption="Source Image")
        show_face(proj.reshape(size), caption="Projected Image")
        show_face(error.reshape(size), caption=caption)
