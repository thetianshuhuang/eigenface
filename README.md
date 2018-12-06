# Eigenface

## Usage
The faces or MNIST datasets are not included in the repository. Place a folder named ```data``` with the faces into a folder named ```data```, test faces in a folder ```testcase```, and the mmnist data ```mnist.mat``` directly into the ```eigenface``` folder.

Run linear PCA with ```python eigenface.py```.

Run kernel PCA eigenfaces with ```python kernel_eigenface.py```. Specify the kernel used with ```python kernel_eigenface.py gaussian``` or ```python kernel_eigenface.py polyonmial```.

Run kernel PCA on the MNIST dataset with ```python eigennumbers.py```. ```polynomial``` and ```gaussian``` can again be used as arguments to specify the kernel.

## Sample Results
