import cv2 as cv
from cv2 import Mat
import numpy as np

def load_image(image_path) -> Mat:
    img = cv.imread(image_path)
    return img

def mirrow_image(img: Mat):
    '''
    Mirror an image
    '''

    # Step 2: get width and height of the image
    height, width = img.shape[0], img.shape[1]
    newImg = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            blue = img[i,j,0]
            green = img[i,j,1]
            red = img[i,j,2]
            # if j < width // 2:
            #     img[i,j] = img[i, width - 1 - j]
            # else :
            #     img[i,j] = [0, 0, 0]
            newRow = i
            newCol = width - 1 - j
            newImg[newRow, newCol] = [blue, green, red]
    return newImg



img = load_image('./img/facetree.jpg')

height, width = img.shape[0], img.shape[1]
print("Image height:", height)
print("Image width:", width)

# img1 = img.copy()
img1 = mirrow_image(img)

cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
cv.moveWindow('Original Image', 0, 0)
cv.imshow('Original Image', img)

cv.namedWindow('Inverse Image', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image', width, 0)
cv.imshow('Inverse Image',img1)

cv.waitKey(0)
cv.destroyAllWindows()