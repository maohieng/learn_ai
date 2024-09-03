#'''
import cv2 as cv

def load_image(image_path):
    img = cv.imread(image_path)
    print("Image shape:", img.shape)
    print("Image height:", img.shape[0])
    print("Image width:", img.shape[1])
    print("Image size:", img.size, "bytes")
    return img

def convert_to_gray_scale(img):
    '''
    converts a color image into a grayscale image
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
            # Step 5: calculate the intensity of the each pixel by using the equation
            # gray = 0.299*red + 0.587*green + 0.114*blue
            newIntensity = 0.299*red + 0.587*green + 0.114*blue
            # Step 6: get intensity for each pixel of new image
            img[i,j] = newIntensity

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

img = load_image('./fruit1.jpg')
height, width, channels = img.shape

# TP01: Make 1/4 red color
img_red = img.copy()
for i in range(int(height/2)):
    for j in range(int(width/2)):
        img_red[i,j,0] = 0
        img_red[i,j,1] = 0
        img_red[i,j,2] = 255

# TP02: Make 1/4 original, 1/4 blue color, 1/4 green color, 1/4 red color
img_color = img.copy()
for i in range(height):
    for j in range(width):
        if i < int(height/2) and j < int(width/2):
            # Original image
            img_color[i,j,0] = img[i,j,0]
            img_color[i,j,1] = img[i,j,1]
            img_color[i,j,2] = img[i,j,2]
        elif i < int(height/2) and j >= int(width/2):
            # Blue color
            img_color[i,j,1] = 0
            img_color[i,j,2] = 0
        elif i >= int(height/2) and j < int(width/2):
            # Green color
            img_color[i,j,0] = 0
            img_color[i,j,2] = 0
        else:
            # Red color
            img_color[i,j,0] = 0
            img_color[i,j,1] = 0

# TP03: Convert to gray scale using calculation
img1 = img.copy()
convert_to_gray_scale(img1)

# Inverse color
inverse_img = img.copy()
inverse_color(inverse_img)


cv.imshow('Original Image', img)

cv.namedWindow('Red Image', cv.WINDOW_NORMAL)
cv.moveWindow('Red Image', width, 0)
cv.imshow('Red Image', img_red)

cv.namedWindow('Color Image', cv.WINDOW_NORMAL)
cv.moveWindow('Color Image', 0, height)
cv.imshow('Color Image', img_color)

cv.namedWindow('Gray Image', cv.WINDOW_NORMAL)
cv.moveWindow('Gray Image', width, height)
cv.imshow('Gray Image',img1)

cv.namedWindow('Inverse Image', cv.WINDOW_NORMAL)
cv.moveWindow('Inverse Image', width+width, 0)
cv.imshow('Inverse Image', inverse_img)

cv.waitKey(0)
cv.destroyAllWindows()
