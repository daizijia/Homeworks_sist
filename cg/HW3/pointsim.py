import numpy as np
import pandas as pd
from skimage import filters, morphology, color
#from triangulared.utils import gaussian_mask, default

def get_triangle_colour(triangles, image, agg_func=np.median):

    # create a list of all pixel coordinates
    ymax, xmax = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(xmax), np.arange(ymax))
    pixel_coords = np.c_[xx.ravel(), yy.ravel()]

    # for each pixel, identify which triangle it belongs to
    triangles_for_coord = triangles.find_simplex(pixel_coords)

    df = pd.DataFrame({
        "triangle": triangles_for_coord,
        "r": image.reshape(-1, 3)[:, 0],
        "g": image.reshape(-1, 3)[:, 1],
        "b": image.reshape(-1, 3)[:, 2]
    })

    n_triangles = triangles.vertices.shape[0]

    by_triangle = (
        df
            .groupby("triangle")
        [["r", "g", "b"]]
            .aggregate(agg_func)
            .reindex(range(n_triangles), fill_value=0)
        # some triangles might not have pixels in them
    )

    return by_triangle.values / 256


def gaussian_mask(x, y, shape, amp=1, sigma=15):
    xv, yv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    g = amp * np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
    return g


def default(value, default_value):
    if value is None:
        return default_value
    return value

def edge_points(image, length_scale=200,
                n_horizontal_points=None,
                n_vertical_points=None):

    ymax, xmax = image.shape[:2]

    if n_horizontal_points is None:
        n_horizontal_points = int(xmax / length_scale)

    if n_vertical_points is None:
        n_vertical_points = int(ymax / length_scale)

    delta_x = xmax / n_horizontal_points
    delta_y = ymax / n_vertical_points

    return np.array(
        [[0, 0], [xmax, 0], [0, ymax], [xmax, ymax]]
        + [[delta_x * i, 0] for i in range(1, n_horizontal_points)]
        + [[delta_x * i, ymax] for i in range(1, n_horizontal_points)]
        + [[0, delta_y * i] for i in range(1, n_vertical_points)]
        + [[xmax, delta_y * i] for i in range(1, n_vertical_points)]
    )


def generate_uniform_random_points(image, n_points=100):
    ymax, xmax = image.shape[:2]
    points = np.random.uniform(size=(n_points, 2))
    points *= np.array([xmax, ymax])
    points = np.concatenate([points, edge_points(image)])
    return points


def generate_max_entropy_points(image, n_points=100,
                                entropy_width=None,
                                filter_width=None,
                                suppression_width=None,
                                suppression_amplitude=None):

    # calculate length scale
    ymax, xmax = image.shape[:2]
    length_scale = np.sqrt(xmax*ymax / n_points)
    entropy_width = length_scale * default(entropy_width, 0.2)
    filter_width = length_scale * default(filter_width, 0.1)
    suppression_width = length_scale * default(suppression_width, 0.3)
    suppression_amplitude = default(suppression_amplitude, 3)

    # convert to grayscale
    im2 = color.rgb2gray(image)

    # filter
    im2 = (
        255 * filters.gaussian(im2, sigma=filter_width, multichannel=True)
    ).astype("uint8")

    # calculate entropy
    im2 = filters.rank.entropy(im2, morphology.disk(entropy_width))

    points = []
    for _ in range(n_points):
        y, x = np.unravel_index(np.argmax(im2), im2.shape)
        im2 -= gaussian_mask(x, y,
                             shape=im2.shape[:2],
                             amp=suppression_amplitude,
                             sigma=suppression_width)
        points.append((x, y))

    points = np.array(points)
    return points