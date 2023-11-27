"""use skimage"""
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

image = img_as_float(io.imread("Datas/xiaokun.png"))
segments = slic(image, n_segments = 500, sigma = 5)#0 10 100 500 1000 10000
print(segments)
print(segments.shape)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))

plt.savefig('Results/result.png')


