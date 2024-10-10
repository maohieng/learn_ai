import cv2

def apply_darkness(image, modify_value=100):
    rows, cols, channels = image.shape
    dark_image = image.copy()
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                new_intensity = image[i, j, k] - modify_value
                dark_image[i, j, k] = max(0, new_intensity)
    return dark_image

def apply_brightness(image, modify_value=100):
    rows, cols, channels = image.shape
    bright_image = image.copy()
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                new_intensity = image[i, j, k] + modify_value
                bright_image[i, j, k] = min(255, new_intensity)
    return bright_image

if __name__ == "__main__":
    image = cv2.imread('facetree.jpg')
    dark_image = apply_darkness(image)
    bright_image = apply_brightness(image)
    cv2.imshow('Dark Image', dark_image)
    cv2.imshow('Bright Image', bright_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
