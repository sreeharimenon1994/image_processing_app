# Application for Image processing

Add images to be processed in "images" folder. There are a lot of filters and mask that can be applied. The app is made with Tkinter and is easy to add more feature.


## UI 
Select required filter and images to be applied to.

![ui1](/demo/ui2.png)


![ui2](/demo/ui1.png)

## Features
* All the results are automatically saved to the results folder.
* There is an ROI checkbox on top left.
* In Re-Scaling the first image is scaled by 1.5 and second by 0.4.
* In Shifting the first image is shifted by a random number and second by a random matrix.
* In Re-Scaling and Shifting, the first image is scaled by 0.7 and the second by 1.8 and both are shifted by a random number.
* In Addition, Subtraction, Multiplication and Division the scaling is by 0.6 and shifted by a random number.
* For ROI-Based Operation, a chess board like image is used and is availbale im images folder by the name "ChessRoi.jpg". Change the variable "RoiFile" to change region.
* For Bit-Plane Slicing, bit = 4.
* For Salt-and-Pepper Noise, a matrix containing just 0 and 255 is used to create noise.
* In Simple Thresholding, a threshold value of 127 is used for first image and 200 for second image.
* For Automated Thresholding, The threshold value is calculated by finding the mean.
* For Adaptive Thresholding, the mean and standard deviation is calculated and all values that are not in between those range are removed.

## Installation

pip:

    pip install -r requirements.txt

## How to Use

To using this repo, some things you should to know:

* To execute run  `python main.py` in src folder.

## Documentation

* https://docs.python.org/3/library/tkinter.html
