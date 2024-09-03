import cv2 as cv

def load_image(image_path):
    img = cv.imread(image_path)
    return img

def bitplane(img):
    height, width = img.shape[0], img.shape[1]
    for i in range(height):
        for j in range(width):
            red = img[i,j,2]
            green = img[i,j,1]
            blue = img[i,j,0]
            img[i,j,2] = red
            img[i,j,1] = green
            img[i,j,0] = blue
        