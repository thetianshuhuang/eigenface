# Eigenface

## Usage
The faces or MNIST datasets are not included in the repository. Place a folder named ```data``` with the faces into a folder named ```data```, test faces in a folder ```testcase```, and the mmnist data ```mnist.mat``` directly into the ```eigenface``` folder.

Run linear PCA with ```python eigenface.py```.

Run kernel PCA eigenfaces with ```python kernel_eigenface.py```. Specify the kernel used with ```python kernel_eigenface.py gaussian``` or ```python kernel_eigenface.py polyonmial```.

Run kernel PCA on the MNIST dataset with ```python eigennumbers.py```. ```polynomial``` and ```gaussian``` can again be used as arguments to specify the kernel.

## Sample Results
Input image:
<img src="https://github.com/thetianshuhuang/eigenface/blob/master/examples/input.png" width="100">

Recovery using Linear PCA, 20 eigenvectors
<img src="https://github.com/thetianshuhuang/eigenface/blob/master/examples/linear_recovery.PNG" width="100">

Recovery using Kernel PCA with a Gaussian Kernel
<img src="https://github.com/thetianshuhuang/eigenface/blob/master/examples/kernel_recovery.PNG" width="100">
