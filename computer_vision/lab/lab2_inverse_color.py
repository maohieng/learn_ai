import cv2 as cv

def load_image(image_path):
    img = cv.imread(image_path)
    return img

def inverse_color(img):
    '''
    inverse the color of an image
    '''
    # Step 2: get width and height of the image
    height, width = img.shape[0], img.shape[1]

    # Step 3: use 2 loops
    for i in range(height): # height
        for j in range(width): # width
            # Step 4: get the RGB values of each pixel
            red = img[i,j,2]
            green = img[i,j,1]
            blue = img[i,j,0]
            # Step 5: calculate the inverse of the color of each pixel
            newRed = 255 - red
            newGreen = 255 - green
            newBlue = 255 - blue
            # Step 6: assign the new RGB values to the pixel
            img[i,j,2] = newRed
            img[i,j,1] = newGreen
            img[i,j,0] = newBlue

img = load_image('./img/fruit4.jpg')

height, width, channels = img.shape
print("Image height:", height)
print("Image width:", width)
print("Image channels:", channels)
print("Image size:", img.size, "bytes")

img1 = img.copy()
inverse_color(img1)

cv.imshow('Original Image', img)

cv.namedWindow('Inverse Image', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image', width, 0)
cv.imshow('Inverse Image',img1)

cv.waitKey(0)
cv.destroyAllWindows()