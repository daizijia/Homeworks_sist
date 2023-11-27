import cv2
import argparse
import numpy as np
from utils import *
from time import time


def timing(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


class SeamCarving:
    def __init__(self):
        # TODO: define our internal variable here if necessary
        pass

    @classmethod
    @timing
    def enlarge(cls, image: np.ndarray, k: int) -> np.ndarray:
        # TODO: enlarge image by k seams
        return image

    @classmethod
    @timing
    def removal(cls, image: np.ndarray, k: int) -> np.ndarray:
        # TODO: remove k seams of the image
        for i in range(k):
            energy = energy_map(image)
            Energy, seam = find_seam(energy)
            image = delete_seam(image, seam, Energy)
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", "-i", type=str, help="input image path", required=True
    )
    parser.add_argument(
        "--output", "-o", type=str, help="output image path", required=True
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="enlarge or removal",
        choices=["enlarge", "removal"],
        required=True,
    )
    parser.add_argument(
        "--num_seams",
        "-n",
        type=int,
        help="number of seams used in algorithm",
        required=True,
    )

    args = parser.parse_args()

    image_data = cv2.imread(args.image_path, cv2.IMREAD_ANYCOLOR)
    algorithm = {"enlarge": SeamCarving.enlarge, "removal": SeamCarving.removal}[
        args.mode
    ]
    output_image = algorithm(image_data, args.num_seams)

    cv2.imwrite(args.output, output_image)
