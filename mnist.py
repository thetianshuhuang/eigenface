
"""MNIST numbers data set

Attributes
----------
NUMBERS : np.array
    Data matrix for the MNIST dataset; 60000x784.
"""

from scipy import io

IMAGES = io.loadmat("mnist.mat")["image_mat"] / 255
