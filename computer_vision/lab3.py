import cv2 as cv
# from cv2 import Mat
from PIL import Image

def load_image(image_path):
    img = cv.imread(image_path)
    return img

def mirrow_image(img):
    '''
    Mirror an image
    '''

    # Step 2: get width and height of the image
    height, width = img.shape[0], img.shape[1]
    for i in range(height):
        for j in range(width):
            if j < width // 2:
                img[i,j] = img[i, width - 1 - j]
            else :
                img[i,j] = [0, 0, 0]

    # Step 3: use 2 loops
    # for i in range(height): # height
    #     # reverse loop through width
    #     for j in range(width):
    #         # Step 4: get the RGB values of each pixel
    #         red = img[i,j,2]
    #         green = img[i,j,1]
    #         blue = img[i,j,0]
    #         # Step 5: calculate the new position of the pixel
    #         new_j = width - 1 - j
    #         # Step 6: assign the pixel to the new position
    #         img[i,new_j,2] = red
    #         img[i,new_j,1] = green
    #         img[i,new_j,0] = blue


def mirrow_by_pil(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

img = load_image('./img/facetree.jpg')

height, width = img.shape[0], img.shape[1]
print("Image height:", height)
print("Image width:", width)

img1 = img.copy()
mirrow_image(img1)

cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
cv.moveWindow('Original Image', 0, 0)
cv.imshow('Original Image', img)

cv.namedWindow('Inverse Image', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image', width, 0)
cv.imshow('Inverse Image',img1)

img2 = mirrow_by_pil(img)
cv.namedWindow('Inverse Image by PIL', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image by PIL', 0, height)
cv.imshow('Inverse Image by PIL', img2)

cv.waitKey(0)
cv.destroyAllWindows()