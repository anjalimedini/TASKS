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

![arj](https://github.com/anjalimedini/TASKS/assets/169042588/b91d7561-3081-4865-bf64-ed37d65f871c)

OUTPUT:

![histo](https://github.com/anjalimedini/TASKS/assets/169042588/cd78cbf7-8bd9-404f-8c0b-9ea383d9907f)


## BOUNDING BOXES 

A bounding box in essence, is a rectangle that surrounds an object, that specifies its position, class(eg: car, person) and confidence(how likely it is to be at that location). Bounding boxes are mainly used in the task of object detection, where the aim is identifying the position and type of multiple objects in the image. 

Required Packages:

`requirments.txt`

`os,csv,PIL`

`pip install Pillow`

Explanation:

1.Import Librabies:

 `os`: Operating system module for file operations.
 
 `csv`: Module for reading and writing CSV files.
 
 `PIL.Image and PIL.ImageDraw`: Modules from the Python Imaging Library (PIL), used for image processing and drawing.

`import os`

`import csv`

`from PIL import Image, ImageDraw`

2.Defining Paths:

`csv_file = "/home/anjali-medini/Downloads/7622202030987_bounding_box.csv"`

`image_dir = "/home/anjali-medini/Downloads/7622202030987"`

`output_dir = "/home/anjali-medini/Downloads/7622202030987_with_boxes"`

`csv_file`: Path to the CSV file containing bounding box information.

`image_dir`: Directory where the images are stored.

`output_dir`: Directory where the output images with bounding boxes will be saved.

3.Creating Output Directory:

`os.makedirs(output_dir, exist_ok=True)`

 This line ensures that the output directory exists. If it doesn't, it creates it.

 4.Functions:
 `def draw_boxes(image, boxes):
 
    draw = ImageDraw.Draw(image)
    
    for box in boxes:
    
        left = int(box['left'])
        
        top = int(box['top'])
        
        right = int(box['right'])
        
        bottom = int(box['bottom'])
        
        draw.rectangle([left, top, right, bottom], outline="blue")
        
    return image

    `def crop_image(image, boxes):
    
    cropped_images = []
    
    for box in boxes:
    
        left = int(box['left'])
        
        top = int(box['top'])
        
        right = int(box['right'])
        
        bottom = int(box['bottom'])
        
        cropped_img = image.crop((left, top, right, bottom))
        
        cropped_images.append(cropped_img)
        
    return cropped_images`

    `draw_boxes(image, boxes)`: Draws bounding boxes on an image. It takes an image and a list of dictionaries (each containing the coordinates of a bounding box) as input and returns the image with bounding boxes drawn on it.

    `crop_image(image, boxes)`: Crops the image based on the provided bounding box coordinates. It returns a list of cropped images.

5.Reading CSV File:

`with open(csv_file, 'r') as file:` Opens the CSV file in read mode.

`csv_reader = csv.DictReader(file)`: Creates a CSV reader object that returns each row as a dictionary.

6.Processing Each Row in the CSV:

     for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))

It iterates over each row in the CSV file.

    `image_name`: Extracts the filename of the image from the current row.
    
    `image_path`: Constructs the full path to the image file.
    
    `output_path`: Constructs the full path where the processed images will be saved.
    
    `image` = Image.open(image_path): Opens the image using PIL.
    
    `boxes`: Extracts the bounding box coordinates from the current row and stores them in a list of dictionaries.

7.Processing Images:

cropped_images: Calls the crop_image function to crop the image based on the bounding box coordinates.

It iterates over the cropped images, saves each one with a prefix indicating its index, and adds it to the output directory.

full_image_with_boxes: Calls the draw_boxes function to draw bounding boxes on the original image.

Saves the image with bounding boxes added to the output directory.

INPUT:


![7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/anjalimedini/TASKS/assets/169042588/4aaa1a75-5903-4a1b-ba80-f0898648057b)

OUTPUT1:

![0_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/anjalimedini/TASKS/assets/169042588/adccef02-9f1b-4a9d-96e2-6a0df4df4b26)

OUTPUT2:

![full_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/anjalimedini/TASKS/assets/169042588/8dc3fe50-ccb5-4e96-bfac-e365f71094a5)

This script essentially reads bounding box information from a CSV file, applies the bounding boxes to the images, and saves the modified images to a specified output directory.




    





 



  

  


 
    



