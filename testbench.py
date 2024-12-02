import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from PIL import Image

from steerablepyramid import SteerablePyramid

if __name__  == "__main__":
    steerable_pyramid = SteerablePyramid(4, 3, [256, 256])
    plt.imshow(steerable_pyramid.h0_filter)
    plt.savefig("artifacts/h0_filter.png")
    plt.imshow(steerable_pyramid.l0_filter)
    plt.savefig("artifacts/l0_filter.png")

    plt.imshow(steerable_pyramid.l_filters[0])
    plt.savefig("artifacts/l1_filter.png")

    plt.imshow(steerable_pyramid.h_filters[0])
    plt.savefig("artifacts/h1_filter.png")

    plt.imshow(steerable_pyramid.b_filters[0][0])
    plt.savefig("artifacts/b1_1_filter.png")

    plt.imshow(steerable_pyramid.b_filters[0][1])
    plt.savefig("artifacts/b1_2_filter.png")

    plt.imshow(steerable_pyramid.b_filters[0][2])
    plt.savefig("artifacts/b1_3_filter.png")

    plt.imshow(steerable_pyramid.b_filters[0][3])
    plt.savefig("artifacts/b1_4_filter.png")

    image = np.asarray(Image.open("artifacts/saucer-mono256.png").convert("L"))
    
    out = steerable_pyramid.create_pyramids(image)
    breakpoint()