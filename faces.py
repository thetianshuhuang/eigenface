
from common import load_images

_IMAGE_NAMES = ["data/{idx}.png".format(idx=str(i + 1)) for i in range(640)]
IMAGES = load_images(_IMAGE_NAMES)

_TESTS = ["testcase/test_01.png", "testcase/test_02.png"]
TEST_IMAGES = load_images(_TESTS)
