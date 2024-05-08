## WEBCAM
A webcam is a digital camera that captures video and audio data and transmits it in real-time over the internet. It is commonly used for video conferencing, live streaming, online meetings, and recording videos.

We will use OpenCV and PyGame libraries.

Using OPENCV:

OpenCV library is compatible with the Linux and windows both operating system. Users need to install the OpenCV library on their local computer using the below command before they go ahead.

```requirments.txt```

```Install command - pip install opencv-python```

APPROACH

1.import opencv library:

```import cv2```

2.define a video capture object:

```vid = cv2.VideoCapture(0) ```

3.loops for capturing the displaying frames:

```while(True):``` 
      
4.Capture the video frame by frame:

```ret, frame = vid.read()```

5.displaying the frame:

```cv2.imshow('frame', frame)```

6. checking for Quit result:

  if cv2.waitKey(1) & 0xFF == ord('q'): 
   
   break

7. After the loop release the cap object, Destroy all the windows: 
   
   ```vid.release()```

  OUTPUT :

[Screencast from 08-05-24 11:47:54 AM IST.webm](https://github.com/anjalimedini/TASKS/assets/169042588/7b398e38-6366-488c-8688-037f9efe0839)



## Iterate The First 10 Numbers

## This will be explain each iteration,print the sum of the current and previous number

1.`num = list(range(10))`

This line creates a list num containing numbers from 0 to 9.using the 'range()' function.

2.`previousnum = 0`

Here, previousnum is initialized to 0. This variable will be used to keep track of the previous number in each iteration of the loop.

3.`for i in num:`

This is a loop that iterates over each element (`i`) in the list `num`.

4.Inside the loop:

`sum = previousnum + 1`

This line calculates the sum of the previous number (`previousnum`) and 1, not the current number (`i`). This is where the mistake is. It should be `sum = previousnum + i` to add the current number to the previous number.

`print('Current number' + str(i)+ 'previous number'+ str(previousnum)+ 'is' + str(sum))`

This line prints the current number (`i`), the previous number (`previousnum`), and their sum (`sum`). However, the formatting of the string is incorrect, it might be better to add some spaces or punctuation to make the output clearer.

`previousnum = i`
This line updates `previousnum` to be equal to the current number `i`, so that in the next iteration of the loop, it will hold the value of the current number as the previous number.

OUTPUT :

Current number0previous number0is1

Current number1previous number0is1

Current number2previous number1is2

Current number3previous number2is3

Current number4previous number3is4

Current number5previous number4is5

Current number6previous number5is6

Current number7previous number6is7

Current number8previous number7is8

Current number9previous number8is9

So, the code attempts to iterate through the list `num`, but it mistakenly calculates the sum of the previous number and 1 instead of the current number and the previous number.


## HISTOGRAM

A histogram is a bar graph-like representation of data that buckets a range of classes into columns along the horizontal x-axis. The vertical y-axis represents the number count or percentage of occurrences in the data for each column. Columns can be used to visualize patterns of data distributions.

1.Required Libraries:

`requirments.txt`

`Install command - pip install opencv-python`

`Install command - pip install matplotlib`

`Install command - pip install numpy`

2.Import Libraries:

`import cv2`

`from matplotlib import pyplot as plt`

`import numpy as np`

3. Read image:

 `img = cv2.imread('/home/anjali-medini/Desktop/image/arj.jpeg', 0)`

 `cv2.imread()`: This function reads an image from the specified file path.

 `/home/anjali-medini/Desktop/image/arj.jpeg`: This is the file path of the image to be read.

 `0`: This argument specifies that the image should be read in grayscale mode. If you omit this argument or use `1` or `-1`, it reads the image in color mode.

4.Calculate Histogram:

 `histr = cv2.calcHist([img], [0], None, [256], [0,256])`

 `cv2.calcHist()`: This function calculates the histogram of an image.
 
`[img]`: This is the input image for which the histogram needs to be calculated.

`[0]`: This specifies the channel for which the histogram is calculated. Since the image is in grayscale, there is only one channel (0).

`None`: This argument is a mask. If provided, the histogram will be calculated only for the pixels where the mask image is non-zero.

`[256]`: This specifies the number of bins for the histogram.

`[0, 256]`: This specifies the range of pixel values. The histogram will be calculated for pixel values in the range [0, 256).

5.Plot Histogram:

`plt.plot(histr)`

`plt.show()`

  `plt.plot(): This function is used to plot the histogram.`
  
 `histr: This is the histogram calculated using cv2.calcHist().`
   
`plt.show(): This function displays the plotted histogram.`

INPUT:

 



  

  


 
    



