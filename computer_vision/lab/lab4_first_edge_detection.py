import cv2
readimg =  cv2.imread('./img/fruit2.jpg')
maskSobelX = [[-1,0,1], [-2,0,2], [-1,0,1]]
maskSobelY = [[1,2,1], [0,0,0], [-1,-2,-1]]

mOutImg = readimg.copy()
#extract height and width
height, width, channel=readimg.shape
print(width,height,channel)

# Temporary Arrays for Sobel Calculations
pTmpXB = (width*height)*[0]
pTmpXG = (width*height)*[0]
pTmpXR = (width*height)*[0]
pTmpYB = (width*height)*[0]
pTmpYG = (width*height)*[0]
pTmpYR = (width*height)*[0]
#initailize image to black or 0
mOutImg = mOutImg*0

# Sobel Filtering
#apply mask to original image
for i in range(1,height-1):
    for j in range(1,width-1):
        newValueBx = 0
        newValueGx = 0
        newValueRx = 0
        newValueBy = 0
        newValueGy = 0
        newValueRy = 0
        # I1(i+k-1,j+l-1)*k(k,l) --> mc=k; mr=l
        for mr in range(3):
            for mc in range(3):
                # I1(i+k-1,j+l-1)
                r,g,b = readimg[i+mc-1,j+mr-1] 
                # K(k,l) = maskSobelX[mr][mc]
                newValueBx += maskSobelX[mr][mc]*b
                newValueGx += maskSobelX[mr][mc]*g
                newValueRx += maskSobelX[mr][mc]*r
                newValueBy += maskSobelY[mr][mc]*b
                newValueGy += maskSobelY[mr][mc]*g
                newValueRy += maskSobelY[mr][mc]*r
        pTmpYB[i*width+j] = newValueBy
        pTmpYG[i*width+j] = newValueGy
        pTmpYR[i*width+j] = newValueRy
        pTmpXB[i*width+j] = newValueBx
        pTmpXG[i*width+j] = newValueGx
        pTmpXR[i*width+j] = newValueRx

# Absolute Values of Sobel Responses
# convert to positive
for i in range(1,height-1):
    for j in range(1,width-1):
        constBVal1,constGVal1,constRVal1=pTmpXB[i*width+j],pTmpXG[i*width+j],pTmpXR[i*width+j]
        constBVal2,constGVal2,constRVal2=pTmpYB[i*width+j],pTmpYG[i*width+j],pTmpYR[i*width+j]
        if constBVal1<0:
            constBVal1 = -constBVal1
        if constGVal1<0:
            constGVal1 = -constGVal1
        if constRVal1<0:
            constRVal1 = -constRVal1
        if constBVal2<0:
            constBVal2 = -constBVal2
        if constGVal2<0:
            constGVal2 = -constGVal2
        if constRVal2<0:
            constRVal2 = -constRVal2
        pTmpXB[i*width+j] = constBVal1+constBVal2
        pTmpXG[i*width+j] = constGVal1+constGVal2
        pTmpXR[i*width+j] = constRVal1+constRVal2
#new max and min of picture
minB=minG=minR=100000000
maxB=maxG=maxR=-100000000
for i in range(1,height-1):
    for j in range(1,width-1):
        newValueB=pTmpXB[i*width+j]
        newValueG=pTmpXG[i*width+j]
        newValueR=pTmpXR[i*width+j]
        if(newValueB<minB):
            minB = newValueB
        if(newValueB>maxB):
            maxB = newValueB
        if(newValueG<minG):
            minG = newValueG
        if(newValueG>maxG): 
            maxG = newValueG
        if(newValueR<minR):
            minR = newValueR
        if(newValueR>maxR):
            maxR = newValueR
# Normalization: 
# normalize number --> can be range from 0 to 1 (original from 0 to 255)
constBVal1 = (float(255.0/(maxB-minB)))
constBVal2 = (float(-255.0*minB/(maxB-minB)))
constGVal1 = (float(255.0/(maxG-minG)))
constGVal2 = (float(-255.0*minG/(maxG-minG)))
constRVal1 = (float(255.0/(maxR-minR)))
constRVal2 = (float(-255.0 * minR/(maxR-minR)))

# Apply Normalized Values and Store in Output Image
for i in range(1,height-1):
    for j in range(1,width-1):
        newValueB = pTmpXB[i*width+j]
        newValueG = pTmpXG[i*width+j]
        newValueR = pTmpXR[i*width+j]
        newValueB = constBVal1*newValueB+constBVal2
        newValueG = constGVal1*newValueG+constGVal2
        newValueR = constRVal1*newValueR+constRVal2
        if newValueB<0:
            newValueB = 0
        if newValueB>255:
            newValueB = 255
        if newValueG<0:
            newValueG = 0
        if newValueG>255:
            newValueG = 255
        if newValueR<0:
            newValueR = 0
        if newValueR>255:
            newValueR = 255
        mOutImg[i,j] = [newValueB,newValueG,newValueR]

pTmpXB = []
pTmpXG = []
pTmpXR = []
pTmpYB = []
pTmpYG = []
pTmpYR = []
#print(mOutImg)
cv2.namedWindow("Origin",cv2.WINDOW_NORMAL)
cv2.moveWindow("Origin",0,0)
cv2.imshow("Origin",readimg)

cv2.namedWindow("Sobel Edge Detection",cv2.WINDOW_NORMAL)
cv2.moveWindow("Sobel Edge Detection",width,0)
cv2.imshow("Sobel Edge Detection",mOutImg)

cv2.waitKey(0)
cv2.destroyAllWindows()