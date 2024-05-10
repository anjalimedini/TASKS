
# importing required libraries of opencv 
import cv2 
  
# importing library for plotting 
from matplotlib import pyplot as plt 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", help = "Enter the path of the image")
args = vars(parser.parse_args())
  
# reads an input image 
img = cv2.imread(args['image_path']) 

  
# find frequency of pixels in range 0-255 
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
  
# show the plotting graph of an image 
plt.plot(histr) 
plt.show() 
