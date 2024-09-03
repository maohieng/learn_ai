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

img = load_image('./img/fruit1.jpg')

height, width, channels = img.shape

img1 = img.copy()
convert_to_gray_scale(img1)

cv.imshow('Original Image', img)

cv.namedWindow('Gray Image', cv.WINDOW_NORMAL)
cv.moveWindow('Gray Image', width, 0)
cv.imshow('Gray Image',img1)

cv.waitKey(0)
cv.destroyAllWindows()