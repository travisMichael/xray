import numpy as np
from PIL import Image

def rotation(array):
    rand = random.randint(1,100)

    if rand >= 0 and rand < 10:
        angle = 10
    elif rand < 20:
        angle = -10
    elif rand < 30:
        angle = 8
    elif rand < 40:
        angle = -8
    elif rand < 50:
        angle = 6
    elif rand < 60:
        angle = -6
    elif rand < 70:
        angle = 4
    elif rand < 80:
        angle = -4
    elif rand < 90:
        angle = 2
    else:
        angle = -2

    image = Image.fromarray(array)
    rotated_image = image.rotate(angle)
    return np.array(rotated_image.getdata()).reshape(224, 224)

