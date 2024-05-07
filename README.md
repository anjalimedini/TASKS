## HISTOGRAM

1.INSTALL FOLLOWING PACKAGES

 ```Opencv,Matplotlib```
 
2.CODE
 ```bash
import cv2 
from matplotlib import pyplot as plt 
img = cv2.imread('/home/anjali-medini/Desktop/image/arj.jpeg',0)
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr)
plt.show() 
```

