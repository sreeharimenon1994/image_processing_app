Run cw_ui.py.
All the results are automatically saved to the results folder.
There is an ROI checkbox on top left.

1. In Re-Scaling the first image is scaled by 1.5 and second by 0.4.
2. In Shifting the first image is shifted by a random number and second by a random matrix.
3. In Re-Scaling and Shifting, the first image is scaled by 0.3 and the second by 1.8 and both are shifted by a random number.
4. In Addition, Subtraction, Multiplication and Division the scaling is by 0.6 and shifted by a random number.
5. For ROI-Based Operation, a chess board like image is used and is availbale im images folder by the name "ChessRoi.jpg".
6. For Bit-Plane Slicing, bit = 4.
7. For Salt-and-Pepper Noise, a matrix containing just 0 and 255 is used to create noise.
8. In Simple Thresholding, a threshold value of 127 is used for first image and 200 for second image.
9. For Automated Thresholding, The threshold value is calculated by finding the mean.
10. For Adaptive Thresholding, the mean and standard deviation is calculated and all values that are not in between those range are removed.
