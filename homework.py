import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder = '/home/utkupolat/Desktop/COMPE464Homework4/images_for_Hw4'
imageFile = list([os.path.join(folder, f) for f in os.listdir(folder)])

imageAutumn = cv2.imread('/home/utkupolat/Desktop/COMPE464Homework4/images_for_Hw4/autumn.jpg', cv2.IMREAD_UNCHANGED)
imagePeppers = cv2.imread('/home/utkupolat/Desktop/COMPE464Homework4/images_for_Hw4/peppers.png', cv2.IMREAD_UNCHANGED)

print('Image Autumn Height,Width and Channel : ', imageAutumn.shape)
print('Image Peppers Height,Width and Channel : ', imagePeppers.shape)

def redChannel(img):
    redChannelImg = img[:,:,2]
    redImg = np.zeros(img.shape)
    redImg[:,:,2] = redChannelImg
    cv2.imwrite('/home/utkupolat/Desktop/COMPE464Homework4/redChannel.png',redImg)
def greenChannel(img):
    greenChannelImg = img[:,:,1]
    greenImg = np.zeros(img.shape)
    greenImg[:,:,1] = greenChannelImg
    cv2.imwrite('/home/utkupolat/Desktop/COMPE464Homework4/greenChannel.png',greenImg)
def blueChannel(img):
    blueChannelImg = img[:,:,0]
    blueImg = np.zeros(img.shape)
    blueImg[:,:,0] = blueChannelImg
    cv2.imwrite('/home/utkupolat/Desktop/COMPE464Homework4/blueChannel.png',blueImg)
def plotRGB(file):
    for i in file:
        img = cv2.imread(i,cv2.COLOR_BGR2RGB)
        color = ('r','g','b')
        plt.figure()
        for idx,col in enumerate(color):
            histr = cv2.calcHist([img],[idx],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
    plt.show()
plotRGB(imageFile)
def plotGray(file):
    for i in file:
        img = cv2.imread(i)
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        grayImgHist = cv2.calcHist([grayImg],[0],None,[256],[0,256])
        plt.plot(grayImgHist)
        plt.show()
plotGray(imageFile)
def thresholdFunction(file,th):
    for i in file:
        img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        (thresh, blackAndWhiteImage) = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        plt.imshow(blackAndWhiteImage)
        plt.show()
thresholdFunction(imageFile,192)
def redToGreen(file):
    for i in enumerate(file):
        img = cv2.imread(i)
        imgTorgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        red = img[:,:,2].copy()
        green = img[:,:,1].copy()
        img[:,:,1] = red
        img[:,:,2] = green
        plt.imshow(img)
        plt.show()
redToGreen(imageFile)
def greenToBlue(file):
    for i in enumerate(file):
        img = cv2.imread(i)
        imgTorgb = cv.cvtColor(img, cv2.COLOR_BGR2RGB)
        blue = img[:,:,0].copy()
        green = img[:,:,1].copy()
        img[:,:,1] = blue
        img[:,:,0] = green
        plt.imshow(img)
        plt.show()
greenToBlue(imageFile)
