import cv2
import numpy as np

def energy_map(img):
    # TODO: calculate the energy map of the image
    H,W,_ = img.shape
    energy = np.zeros((H,W))
    
    for C in cv2.split(img):
        temp = np.absolute(cv2.Sobel(C, -1, 1, 0)) + np.absolute(cv2.Sobel(C, -1, 0, 1))
        energy += temp
    return energy

def find_seam(energy):
    # TODO: use dynamic programming to find min seam
    H, W = energy.shape
    seam = np.zeros(energy.shape)
    
    for i in range(1, H):
        for j in range(0, W):
            if j == 0:
                min_index = np.argmin(energy[i - 1, j:j + 1]) + j
                energy[i, j] += int(energy[i - 1, min_index])
                seam[i, j] = min_index
            else:
                min_index = np.argmin(energy[i - 1, j - 1:j + 1]) + j - 1
                energy[i, j] += int(energy[i - 1, min_index])
                seam[i, j] = min_index
    return energy, seam


def delete_seam(image, seam, energy):
    # TODO: delete min seam
    h, w, _ = image.shape
    output = np.zeros((h, w - 1, 3))

    j = np.argmin(energy[-1])
    for i in range(h - 1, 0, -1):
        for k in range(3):
            output[i, :, k] = np.delete(image[i, :, k], [j])    
            j = int(seam[i][j])
    return output