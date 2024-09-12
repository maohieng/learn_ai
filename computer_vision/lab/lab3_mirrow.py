import cv2 as cv
from cv2 import Mat
import numpy as np

def load_image(image_path) -> Mat:
    img = cv.imread(image_path)
    return img

def vertical_mirrow(img):
    '''
    Mirror an image
    '''
    # Step 2: get width and height of the image
    height, width = img.shape[0], img.shape[1]
    newImag = img.copy()
    for i in range(height):
        for j in range(width):
            newImag[i,j] = img[i, width - 1 - j]
    
    return newImag

def horizontal_mirrow(img):
    '''
    Mirror an image
    '''
    # Step 2: get width and height of the image
    height, width = img.shape[0], img.shape[1]
    newImag = img.copy()
    for i in range(height):
        for j in range(width):
            newImag[i,j] = img[-(i+1), j]
    
    return newImag

def both_mirrow(img):
    height, width = img.shape[0], img.shape[1]
    newImag = img.copy()
    for i in range(height):
        for j in range(width):
            newImag[i,j] = img[-(i+1), -(j+1)]
    
    return newImag

img = load_image('./img/facetree.jpg')

height, width = img.shape[0], img.shape[1]
print("Image height:", height)
print("Image width:", width)


cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
cv.moveWindow('Original Image', 0, 0)
cv.imshow('Original Image', img)

img1 = vertical_mirrow(img)
cv.namedWindow('Inverse Image', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image', width, 0)
cv.imshow('Inverse Image',img1)

img2 = horizontal_mirrow(img)
cv.namedWindow('Inverse Image 2', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image 2', 0, height)
cv.imshow('Inverse Image 2',img2)

img3 = both_mirrow(img)
cv.namedWindow('Inverse Image 3', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image 3', width, height)
cv.imshow('Inverse Image 3',img3)

cv.waitKey(0)
cv.destroyAllWindows()