import numpy as np
import cv2


def salt_and_pepper(image):
    image_zeroes = np.zeros(image.shape, np.uint8)
    mean = 0
    sigma = 10
    cv2.randn(image_zeroes,mean,sigma)
    return cv2.add(image, image_zeroes)

