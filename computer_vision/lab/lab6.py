'''
Summary:
This code performs histogram equalization on an image by:

Building the histograms of the Blue, Green, and Red color channels.
Computing the cumulative distribution function (CDF) for each channel.
Using the CDF to create a Look-Up Table (LUT) that remaps the pixel intensity values to equalize the histogram.
The result is displayed alongside the original image, where the equalized image should have enhanced contrast.
'''

from math import floor
import cv2
img = cv2.imread('img/glass.jpg')
copy_img = img.copy()
height,width,kernel = copy_img.shape

histoB=(256)*[0]
histoG=(256)*[0]
histoR=(256)*[0]

LUTB=(256)*[0]
LUTG=(256)*[0]
LUTR=(256)*[0]

for i in range(height):
    for j in range(width):
        r,g,b = img[i,j]
        histoR[r]+=1
        histoG[g]+=1
        histoB[b]+=1

# Cumulative Distribution Function (CDF) Calculation
# Build histogram original
for i in range(256):
    #change if i == 0 so we give histogram equal to its index at 0
    if i == 0:
        histoB[i]=histoB[0]
        histoG[i]=histoG[0]
        histoR[i]=histoR[0]
    else:
        #check the rest calculate like take current index + index-1
        histoB[i]=histoB[i]+histoB[i-1]
        histoG[i]=histoG[i]+histoG[i-1]
        histoR[i]=histoR[i]+histoR[i-1]
        
# Histogram Transformation
# The code transforms the histograms into a mapping for the Look-Up Tables (LUTs).
for i in range(256):
    LUTB[i] = (float(histoB[i])/float(width*height))*255
    LUTG[i] = (float(histoG[i])/float(width*height))*255
    LUTR[i] = (float(histoR[i])/float(width*height))*255


# Apply the LUT (Histogram Equalization)
# For each pixel, the original intensity values (R, G, B) 
# are replaced by the new values from the LUTs (LUTB, LUTG, LUTR), 
# which represent the equalized values.
for i in range(height):
    for j in range(width):
        b,g,r = img[i,j]
        copy_img[i,j] = LUTB[b],LUTG[g],LUTR[r]
        

cv2.imshow('Histogram',copy_img)
cv2.imshow("Original",img)
cv2.waitKey()