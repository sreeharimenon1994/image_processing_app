import tkinter as tk
import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt
import matplotlib._pylab_helpers
from PIL import Image, ImageTk
import os

if not os.path.exists('results'):
    os.makedirs('results')

imgDim = 275
# file for ROI operation. Expected to be inside "images" folder
RoiFile = 'ChessRoi.jpg'

# lookup table for logarithmic operation
LUTLog = {}
tmp = 255/np.log(256)
for x in range(256):
    LUTLog[x] = np.log(1+x)*tmp

# lookup table for power-law
LUTPower = {}
gamma = .01
tmp = 255**(1 - gamma)
for x in range(256):
    LUTPower[x] = tmp*x**gamma

#  random lookup table
RandLUT = {}
for x in range(256):
    RandLUT[x] = np.random.randint(256)

# bit for bit-plane slicing
bit = 4

kernel = {
    "averaging": np.array([[1,1,1],[1,1,1],[1,1,1]])/9,
    "weighted_averaging": np.array([[1,2,1],[2,4,2],[1,2,1]])/16,
    "4_neighbour_laplacian": np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
    "8_neighbour_laplacian": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    "4_neighbour_laplacian_enhancement": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    "8_neighbour_laplacian_enhancement": np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]),
    "roberts": np.array([[0,0,0],[0,0,-1],[0,1,0]]),
    "roberts_with_absolute_value": np.array([[0,0,0],[0,-1,0],[0,0,1]]),
    "sobel_x": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    "sobel_y": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
    "gaussian": np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273,
    "laplacian_of_gaussian": np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
}

#  used to clip all values exceeding the limit
def ClippingImage(imgTmp, norm=False):
    if norm:
        imgTmp = ((imgTmp - np.min(imgTmp))/np.max(imgTmp))*255
    imgTmp[imgTmp < 0] = 0
    imgTmp[imgTmp > 255] = 255
    imgTmp = imgTmp.astype("uint8")
    return imgTmp

# Function to read images in all formats
def ReadImage(path):
    if path.endswith('.raw'):
        imgTmp = np.fromfile(path, dtype='int8', sep="")
        #  finding the dimension to resize(only for square images)
        s = int(math.sqrt(imgTmp.shape[0]))
        imgTmp = imgTmp.reshape(s,s)
        original = cv.cvtColor(imgTmp.astype("uint8"), cv.COLOR_GRAY2BGR)
    else:
        original = cv.imread(path)
        imgTmp = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])
    imgTmp = ClippingImage(imgTmp)
    return original, imgTmp


def RescaleImage(imgTmp, scale):
    imgTmp = imgTmp*scale
    imgTmp = ClippingImage(imgTmp)
    return imgTmp


def ShiftingImage(imgTmp, shift=0):
    if shift == 'all_random':
        shift = np.random.randint(-50,50,imgTmp.shape)
    else:
        shift = np.random.randint(-50,50)
    imgTmp = imgTmp + shift
    imgTmp = ClippingImage(imgTmp)
    return imgTmp


def ArithmeticOperations(a, b, op):
    if op == 'addition':
        res = a + b
    elif op == 'subtraction':
        res = a - b
    elif op == 'multiplication':
        res = a * b
    elif op == 'division':
        res = a / b
    res = ClippingImage(res)
    return res


def BinaryNot(imgTmp):
    imgTmp = np.invert(imgTmp)
    return imgTmp


def BinaryOperations(a, b, op):
    if op == 'and':
        res = np.bitwise_and(a, b)
    elif op == 'or':
        res = np.bitwise_or(a, b)
    elif op == 'xor':
        res = np.bitwise_xor(a, b)
    res = ClippingImage(res)
    return res

# these vectors are created so as to apply function to each item in array
LogLUT = lambda i: LUTLog[i]
LogLUTVector = np.vectorize(LogLUT)
PowerLUT = lambda i: LUTPower[i]
PowerLUTVector = np.vectorize(PowerLUT)
LUTRand = lambda i: RandLUT[i]
RandLUTVector = np.vectorize(LUTRand)
BinaryExtraction = lambda i,j: np.binary_repr(i, width=8)[7-j]
BinaryReprVector = np.vectorize(BinaryExtraction)


def ImageTransformation(imgTmp, op):
    if op == 'negative_linear':
        res = 255 - imgTmp
    elif op == 'logarithmic':
        tmp = LogLUTVector(imgTmp)
        res = tmp*np.log(1+imgTmp)
    elif op == 'power_law':
        tmp = PowerLUTVector(imgTmp)
        res = tmp*imgTmp**gamma
    elif op == 'random_look_up_table':
        res = RandLUTVector(imgTmp)
    elif op == 'bit_plane_slicing':
        res = BinaryReprVector(imgTmp, bit)
        res = res.astype('uint8')
    res = ClippingImage(res, True)
    return res


def Histogram(imgTmp, norm=False):
    #  counting the values in the array, return the index and count
    ind, cnt = np.unique(imgTmp, return_counts=True)
    hist = []
    for x in range(256):
        tmp = np.where(ind==x)[0]
        if len(tmp) == 1:
            tmp = tmp[0]
            hist.append(cnt[tmp])
        else:
            hist.append(0)
    if norm:
        hist = hist/np.sum(cnt)
    return hist


def Convolution(img, kernel):
    dst = (cv.filter2D(np.float32(img), -1, kernel)).astype("uint8")
    return dst

#  created sub matrices of 3x3
def SubMatrices(mat):
    sub_shape = (3, 3)
    view_shape = tuple(np.subtract(mat.shape, sub_shape) + 1) + sub_shape
    strides = mat.strides + mat.strides
    sub_matrices = np.lib.stride_tricks.as_strided(mat, view_shape, strides)
    return sub_matrices


midPoint = lambda x: int((min(x)+max(x))/2)


def Threshold(imgTmp2, limit=[]):
    imgTmp = np.array(imgTmp2, copy=True)
    if len(limit) == 2:
        imgTmp[imgTmp < limit[0]] = 0
        imgTmp[imgTmp > limit[1]] = 0
    elif len(limit) == 1:
        imgTmp[imgTmp < limit[0]] = 0
    imgTmp[imgTmp > 0] = 255
    return imgTmp

#  To convert the matplot figure to an numpy array
def MatplotLibToImage(hist):
    plt.bar(list(range(1,257)), hist)
    fig = plt.figure()
    fig = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()[0].canvas.figure
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")
    return data


def ImageResize(imgTmp, size=()):
    flag = False
    if len(size) == 0:
        size = (imgDim, imgDim)
    testShape = imgTmp.shape
    if len(testShape) == 3 :
        dimTmp = True
    else:
        dimTmp = False
    if testShape[0] != testShape[1]:
        flag = True
        testShape = int((size[1]/testShape[1])*testShape[0])
        size = (imgDim, testShape)

    imgTmp = cv.resize(imgTmp, size, interpolation=cv.INTER_LINEAR)
    if flag:
        tmp = imgDim - testShape
        #  concatenating array with value 240 to prevent image from stretching and 
        #  to get a transparent effect (*tested in windows)
        #  3 dim in case of histograms
        if dimTmp:
            tmp = np.ones([tmp, imgDim, 3], dtype=np.uint8) * 240    
        else:
            tmp = np.ones([tmp, imgDim], dtype=np.uint8) * 240
        imgTmp = np.concatenate((imgTmp, tmp))
        flag = False
    return imgTmp


window = tk.Tk()
window.geometry("900x700")
window.minsize(900, 700)
window.wm_title("Image Processing Coursework")

# imageTkOptions = ["Baboon.bmp", "Cameraman.tif", "Peppers.raw", "Goldhill.tif", "Mars.jpg", "Ocr.png", "WindFarm.png"]

imageTkOptions = list(os.walk('images/'))[0][2]
imageTkOptions = list(filter(lambda x: x!=RoiFile, imageTkOptions))


labTkOptions = {"Lab 1": ["Read Multiple Images"],
                "Lab 2": ["Re-Scaling", "Shifting", "Re-Scaling and Shifting"],
                "Lab 3": ["Addition", "Subtraction", "Multiplication", "Division", "Bitwise NOT", "Bitwise AND", "Bitwise OR", "Bitwise XOR", "ROI-Based Operation"],
                "Lab 4": ["Negative Linear Transform", "Logarithmic Function", "Power-Law", "Random Look-up Table", "Bit-plane Slicing"],
                "Lab 5": ["Histogram", "Normalized Histogram", "Histogram Equalisation"],
                "Lab 6": ["Averaging", "Weighted Averaging", "4-neighbour Laplacian", "8-neighbour Laplacian", "4-neighbour Laplacian Enhancement", "8-neighbour Laplacian Enhancement", "Roberts", "Roberts with Absolute Value", "Sobel X", "Sobel Y", "Gaussian", "Laplacian of Gaussian"],
                "Lab 7": ["Salt-and-Pepper Noise", "Min Filtering", "Max Filtering", "Midpoint Filtering", "Median Filtering"],
                "Lab 8": ["Simple Thresholding", "Automated Thresholding", "Adaptive Thresholding"]
                }
labTkOptions['All Labs'] = [x for sub in labTkOptions.values() for x in sub]

saveasTkOptions = [".bmp", ".jpg", ".png"]


image1TkVar = tk.StringVar(window)
image1TkVar.set("Baboon.bmp")
image2TkVar = tk.StringVar(window)
image2TkVar.set("Cameraman.tif")

labTkVar = tk.StringVar(window)
labTkVar.set("Lab 2")

labxTkVar = tk.StringVar(window)
labxTkVar.set(labTkOptions[labTkVar.get()][0])

saveasTkVar = tk.StringVar(window)
saveasTkVar.set(saveasTkOptions[1])

RoiCheckTkVar = tk.BooleanVar(window)
RoiCheckTkVar.set(False)
OriginalRoi, RoiChess = ReadImage("images/"+RoiFile)
shapeTmp = RoiChess.shape
RoiChess = RoiChess[0:shapeTmp[0], 0:shapeTmp[0]]
RoiChessRe =ImageResize(RoiChess)
RoiChessReNot = BinaryNot(RoiChessRe)

image1TkVarFilter = tk.OptionMenu(window, image1TkVar, *imageTkOptions)
image1TkVarFilter.config(width=13)
image1TkVarFilter.place(x=0.0, y=0.0, anchor=tk.NW)

image2TkVarFilter = tk.OptionMenu(window, image2TkVar, *imageTkOptions)
image2TkVarFilter.config(width=13)
image2TkVarFilter.place(x=123, y=0.0, anchor=tk.NW)

labTkVarFilter = tk.OptionMenu(window, labTkVar, *labTkOptions.keys())
labTkVarFilter.place(relx=0.5, y=0.0, anchor=tk.NE)

labxTkVarFilter = tk.OptionMenu(window, labxTkVar, *labTkOptions[labTkVar.get()])
labxTkVarFilter.place(relx=0.5, y=0.0, anchor=tk.NW)

saveasTkVarFilter = tk.OptionMenu(window, saveasTkVar, *saveasTkOptions)
saveasTkVarFilter.config(width=4)
saveasTkVarFilter.place(relx=.92, y=0.0, anchor=tk.NW)

canvas1 = tk.Canvas(window, width=imgDim, height=imgDim)
canvas2 = tk.Canvas(window, width=imgDim, height=imgDim)
canvas3 = tk.Canvas(window, width=imgDim, height=imgDim)
canvas4 = tk.Canvas(window, width=imgDim, height=imgDim)
canvas5 = tk.Canvas(window, width=imgDim, height=imgDim)

text = tk.Text(window)
text.insert(tk.INSERT, "Save As:")
text.place(relx=.92, y=5, anchor=tk.NE)
text.config(width=8, height=1.3, state="disabled", pady=2.5, bd=0, bg=window.cget("background"))


textInput = tk.Text(window)
textInput.insert(tk.INSERT, "Inputs -")
textInput.place(relx=.03, rely=.25, anchor=tk.W)
textInput.config(width=10, height=1.3, state="disabled", pady=2.5, bd=0, bg=window.cget("background"))


textResult = tk.Text(window)
textResult.insert(tk.INSERT, "Result -")
textResult.place(relx=.03, rely=.70, anchor=tk.W)
textResult.config(width=10, height=1.3, state="disabled", pady=2.5, bd=0, bg=window.cget("background"))

checkbox = tk.Checkbutton(window, text='ROI - Stripe', variable=RoiCheckTkVar, onvalue=True, offvalue=False)
checkbox.place(relx=0.0, y=43, anchor=tk.W)


def DisplayInitImages(img1, img2):
    img1Tmp = ImageResize(img1)
    img2Tmp = ImageResize(img2)

    img1Tmp =  ImageTk.PhotoImage(image=Image.fromarray(img1Tmp))
    img2Tmp =  ImageTk.PhotoImage(image=Image.fromarray(img2Tmp))

    canvas1.create_image(0, 0, anchor="nw", image=img1Tmp)
    canvas1.place(relx=0.2, rely=0.08, anchor=tk.NW)

    canvas2.create_image(0, 0, anchor="nw", image=img2Tmp)
    canvas2.place(relx=0.9, rely=0.08, anchor=tk.NE)

    label1 = tk.Label(window, image=img1Tmp)
    label1.image = img1Tmp

    label2 = tk.Label(window, image=img2Tmp)
    label2.image = img2Tmp


def Result(img1, img2):
    single = False
    labxOp = labxTkVar.get()
    resultImage2 = None
    if labxOp == "Re-Scaling":
        resultImage1 = RescaleImage(img1, 1.5)
        resultImage2 = RescaleImage(img2, .4)
    elif labxOp == "Shifting":
        resultImage1 = ShiftingImage(img1)
        resultImage2 = ShiftingImage(img2, "all_random")
    elif labxOp == "Re-Scaling and Shifting":
        resultImage1 = RescaleImage(img1, .3)
        resultImage2 = RescaleImage(img2, 1.8)
        resultImage1 = ShiftingImage(resultImage1)
        resultImage2 = ShiftingImage(resultImage2)
    elif labxOp == "Addition":
        single = True
        resultImage1 = ArithmeticOperations(img1, img2, 'addition')
        resultImage1 = RescaleImage(resultImage1, .6)
        resultImage1 = ShiftingImage(resultImage1)
    elif labxOp == "Subtraction":
        single = True
        resultImage1 = ArithmeticOperations(img1, img2, 'subtraction')
        resultImage1 = RescaleImage(resultImage1, .6)
        resultImage1 = ShiftingImage(resultImage1)
    elif labxOp == "Multiplication":
        single = True
        resultImage1 = ArithmeticOperations(img1, img2, 'multiplication')
        resultImage1 = RescaleImage(resultImage1, .6)
        resultImage1 = ShiftingImage(resultImage1)
    elif labxOp == "Division":
        single = True
        resultImage1 = ArithmeticOperations(img1, img2, 'division')
        resultImage1 = RescaleImage(resultImage1, .6)
        resultImage1 = ShiftingImage(resultImage1)
    elif labxOp == "Bitwise NOT":
        resultImage1 = BinaryNot(img1)
        resultImage2 = BinaryNot(img2)
    elif labxOp == "Bitwise AND":
        single = True
        resultImage1 = BinaryOperations(img1, img2, 'and')
    elif labxOp == "Bitwise OR":
        single = True
        resultImage1 = BinaryOperations(img1, img2, 'or')
    elif labxOp == "Bitwise XOR":
        single = True
        resultImage1 = BinaryOperations(img1, img2, 'xor')
    elif labxOp == "ROI-Based Operation":
        roi =ImageResize(RoiChess, img1.shape)
        resultImage1 = BinaryOperations(img1, roi, 'and')
        roi =ImageResize(RoiChess, img2.shape)
        resultImage2 = BinaryOperations(img2, roi, 'and')
    elif labxOp == "Negative Linear Transform":
        resultImage1 = ImageTransformation(img1, 'negative_linear')
        resultImage2 = ImageTransformation(img2, 'negative_linear')
    elif labxOp == "Logarithmic Function":
        resultImage1 = ImageTransformation(img1, 'logarithmic')
        resultImage2 = ImageTransformation(img2, 'logarithmic')
    elif labxOp == "Power-Law":
        resultImage1 = ImageTransformation(img1, 'power_law')
        resultImage2 = ImageTransformation(img2, 'power_law')
    elif labxOp == "Random Look-up Table":
        resultImage1 = ImageTransformation(img1, 'random_look_up_table')
        resultImage2 = ImageTransformation(img2, 'random_look_up_table')
    elif labxOp == "Bit-plane Slicing":
        resultImage1 = ImageTransformation(img1, 'bit_plane_slicing')
        resultImage2 = ImageTransformation(img2, 'bit_plane_slicing')
    elif labxOp == "Histogram":
        rng = list(range(1,257))
        hist = Histogram(img1)
        resultImage1 = MatplotLibToImage(hist)
        hist = Histogram(img2)
        resultImage2 = MatplotLibToImage(hist)
    elif labxOp == "Normalized Histogram":
        rng = list(range(1,257))
        hist = Histogram(img1, True)
        resultImage1 = MatplotLibToImage(hist)
        hist = Histogram(img2, True)
        resultImage2 = MatplotLibToImage(hist)
    elif labxOp == "Histogram Equalisation":
        resultImage1 = cv.cvtColor(img1, cv.COLOR_BGR2YUV)
        resultImage1[:,:,0] = cv.equalizeHist(resultImage1[:,:,0])
        resultImage1 = cv.cvtColor(resultImage1, cv.COLOR_YUV2BGR)
        resultImage2 = cv.cvtColor(img2, cv.COLOR_BGR2YUV)
        resultImage2[:,:,0] = cv.equalizeHist(resultImage2[:,:,0])
        resultImage2 = cv.cvtColor(resultImage2, cv.COLOR_YUV2BGR)
    elif labxOp == "Averaging":
        resultImage1 = Convolution(img1, kernel["averaging"])
        resultImage2 = Convolution(img2, kernel["averaging"])
    elif labxOp == "Weighted Averaging":
        resultImage1 = Convolution(img1, kernel["weighted_averaging"])
        resultImage2 = Convolution(img2, kernel["weighted_averaging"])
    elif labxOp == "4-neighbour Laplacian":
        resultImage1 = Convolution(img1, kernel["4_neighbour_laplacian"])
        resultImage2 = Convolution(img2, kernel["4_neighbour_laplacian"])
    elif labxOp == "8-neighbour Laplacian":
        resultImage1 = Convolution(img1, kernel["8_neighbour_laplacian"])
        resultImage2 = Convolution(img2, kernel["8_neighbour_laplacian"])
    elif labxOp == "4-neighbour Laplacian Enhancement":
        resultImage1 = Convolution(img1, kernel["4_neighbour_laplacian_enhancement"])
        resultImage2 = Convolution(img2, kernel["4_neighbour_laplacian_enhancement"])
    elif labxOp == "8-neighbour Laplacian Enhancement":
        resultImage1 = Convolution(img1, kernel["8_neighbour_laplacian_enhancement"])
        resultImage2 = Convolution(img2, kernel["8_neighbour_laplacian_enhancement"])
    elif labxOp == "Roberts":
        resultImage1 = Convolution(img1, kernel["roberts"])
        resultImage2 = Convolution(img2, kernel["roberts"])
    elif labxOp == "Roberts with Absolute Value":
        resultImage1 = Convolution(img1, kernel["roberts_with_absolute_value"])
        resultImage2 = Convolution(img2, kernel["roberts_with_absolute_value"])
    elif labxOp == "Sobel X":
        resultImage1 = Convolution(img1, kernel["sobel_x"])
        resultImage2 = Convolution(img2, kernel["sobel_x"])
    elif labxOp == "Sobel Y":
        resultImage1 = Convolution(img1, kernel["sobel_y"])
        resultImage2 = Convolution(img2, kernel["sobel_y"])
    elif labxOp == "Gaussian":
        resultImage1 = Convolution(img1, kernel["gaussian"])
        resultImage2 = Convolution(img2, kernel["gaussian"])
    elif labxOp == "Laplacian of Gaussian":
        resultImage1 = Convolution(img1, kernel["laplacian_of_gaussian"])
        resultImage2 = Convolution(img2, kernel["laplacian_of_gaussian"])
    elif labxOp == "Salt-and-Pepper Noise":
        noiseSalt = np.random.choice([0,255], img1.shape)
        noisePepper = np.random.choice([0,-255], img1.shape)
        resultImage1 = ClippingImage(img1 + noiseSalt + noisePepper)
        noiseSalt = np.random.choice([0,255], img2.shape)
        noisePepper = np.random.choice([0,-255], img2.shape)
        resultImage2 = ClippingImage(img2 + noiseSalt + noisePepper)
    elif labxOp == "Min Filtering":
        noiseSalt = np.random.choice([0,255], img1.shape)
        noisePepper = np.random.choice([0,-255], img1.shape)
        imgTmp = ClippingImage(img1 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9) 
        sub = np.apply_along_axis(np.min, 2, sub)
        resultImage1 = ClippingImage(sub.reshape(shape[0], shape[1]))# end of img1
        noiseSalt = np.random.choice([0,255], img2.shape)
        noisePepper = np.random.choice([0,-255], img2.shape)
        imgTmp = ClippingImage(img2 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9)
        sub = np.apply_along_axis(np.min, 2, sub)
        resultImage2 = ClippingImage(sub.reshape(shape[0], shape[1]))
    elif labxOp == "Max Filtering":
        noiseSalt = np.random.choice([0,255], img1.shape)
        noisePepper = np.random.choice([0,-255], img1.shape)
        imgTmp = ClippingImage(img1 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9) 
        sub = np.apply_along_axis(np.max, 2, sub)
        resultImage1 = ClippingImage(sub.reshape(shape[0], shape[1]))# end of img1
        noiseSalt = np.random.choice([0,255], img2.shape)
        noisePepper = np.random.choice([0,-255], img2.shape)
        imgTmp = ClippingImage(img2 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9)
        sub = np.apply_along_axis(np.max, 2, sub)
        resultImage2 = ClippingImage(sub.reshape(shape[0], shape[1]))
    elif labxOp == "Midpoint Filtering":
        noiseSalt = np.random.choice([0,255], img1.shape)
        noisePepper = np.random.choice([0,-255], img1.shape)
        imgTmp = ClippingImage(img1 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9) 
        sub = np.apply_along_axis(midPoint, 2, sub)
        resultImage1 = ClippingImage(sub.reshape(shape[0], shape[1]))# end of img1
        noiseSalt = np.random.choice([0,255], img2.shape)
        noisePepper = np.random.choice([0,-255], img2.shape)
        imgTmp = ClippingImage(img2 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9)
        sub = np.apply_along_axis(midPoint, 2, sub)
        resultImage2 = ClippingImage(sub.reshape(shape[0], shape[1]))
    elif labxOp == "Median Filtering":
        noiseSalt = np.random.choice([0,255], img1.shape)
        noisePepper = np.random.choice([0,-255], img1.shape)
        imgTmp = ClippingImage(img1 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9) 
        sub = np.apply_along_axis(np.median, 2, sub)
        resultImage1 = ClippingImage(sub.reshape(shape[0], shape[1]))# end of img1
        noiseSalt = np.random.choice([0,255], img2.shape)
        noisePepper = np.random.choice([0,-255], img2.shape)
        imgTmp = ClippingImage(img2 + noiseSalt + noisePepper)
        sub = SubMatrices(imgTmp)
        shape = sub.shape
        totDim = shape[0] * shape[1]
        sub = sub.reshape(1, totDim, 9)
        sub = np.apply_along_axis(np.median, 2, sub)
        resultImage2 = ClippingImage(sub.reshape(shape[0], shape[1]))
    elif labxOp == "Simple Thresholding":
        resultImage1 = Threshold(img1, [127])
        resultImage2 = Threshold(img2, [200]) 
    elif labxOp == "Automated Thresholding":
        mean = int(np.mean(img1))
        resultImage1 = Threshold(img1, [mean])
        hist = Histogram(img2)
        mean = int(np.mean(img2))
        resultImage2 = Threshold(img2, [mean])
    elif labxOp == "Adaptive Thresholding":
        mean = int(np.mean(img1))
        std = int(np.std(img1))
        resultImage1 = Threshold(img1, [mean - std, mean + std])
        mean = int(np.mean(img2))
        std = int(np.std(img1))
        resultImage2 = Threshold(img2, [mean - std, mean + std])
    return resultImage1, resultImage2, single


def DisplayResultImages(resultImage1, resultImage2, single):
    if single:
        resultImage1 =  ImageTk.PhotoImage(image=Image.fromarray(resultImage1))
        canvas5.create_image(0, 0, anchor="nw", image=resultImage1)
        canvas5.place(relx=0.5, rely=0.53, anchor=tk.N)

        label3 = tk.Label(window, image=resultImage1)
        label3.image = resultImage1

        canvas3.place(relx=0.0, rely=0.0, anchor=tk.SE)
        canvas4.place(relx=0.0, rely=0.0, anchor=tk.SE)
    else:
        resultImage1 =  ImageTk.PhotoImage(image=Image.fromarray(resultImage1))
        resultImage2 =  ImageTk.PhotoImage(image=Image.fromarray(resultImage2))

        canvas3.create_image(0, 0, anchor="nw", image=resultImage1)
        canvas3.place(relx=0.2, rely=0.53, anchor=tk.NW)

        label3 = tk.Label(window, image=resultImage1)
        label3.image = resultImage1

        canvas4.create_image(0, 0, anchor="nw", image=resultImage2)
        canvas4.place(relx=0.9, rely=0.53, anchor=tk.NE)

        label3 = tk.Label(window, image=resultImage2)
        label3.image = resultImage2

        canvas5.place(relx=0.0, rely=0.0, anchor=tk.SE)


def RefreshFrame():
    original1, img1 = ReadImage("images/"+image1TkVar.get())
    original2, img2 = ReadImage("images/"+image2TkVar.get())
    tmp = labTkVar.get()
    if tmp == "Lab 3":
        if img1.shape != img2.shape:
            img1 = ImageResize(img1, (275, 275))
            img2 = ImageResize(img2, (275, 275))
    if tmp == "Lab 5" or tmp == "Lab 1":
        checkbox.place(relx=0.0, y=0.0, anchor=tk.E)
    else:
        checkbox.place(relx=0.0, y=43, anchor=tk.W)
    tmp = labxTkVar.get()
    if tmp == "Histogram Equalisation":
        DisplayInitImages(original1, original2)
        resultImage1, resultImage2, single = Result(original1, original2)
    elif tmp == "Read Multiple Images":
        DisplayInitImages(original1, original2)
        resultImage1, resultImage2, single = original1, original2, False
    else:
        DisplayInitImages(img1, img2)
        resultImage1, resultImage2, single = Result(img1, img2)
    
    resPath = "results/"+labTkVar.get()+" - "+labxTkVar.get()+" - 1"+saveasTkVar.get()
    cv.imwrite(resPath, resultImage1)
    resultImage1 = ImageResize(resultImage1)
    if not single:
        resPath = "results/"+labTkVar.get()+" - "+labxTkVar.get()+" - 2"+saveasTkVar.get()
        cv.imwrite(resPath, resultImage2)
        resultImage2 = ImageResize(resultImage2)

    if RoiCheckTkVar.get() and labTkVar.get() != "Lab 5":
        imgTmp = ImageResize(img1)
        img1 = BinaryOperations(imgTmp, RoiChessReNot, 'and')
        resultImage1 = BinaryOperations(resultImage1, RoiChessRe, 'and') + img1
        if not single:
            imgTmp = ImageResize(img2)
            img2 = BinaryOperations(imgTmp, RoiChessReNot, 'and')
            resultImage2 = BinaryOperations(resultImage2, RoiChessRe, 'and') + img2

    DisplayResultImages(resultImage1, resultImage2, single)


def ChangeDropdown(*args):
    if args[0] == 'PY_VAR2':
        labxTkVar.set(labTkOptions[labTkVar.get()][0])
        labxTkVarFilter['menu'].delete(0, 'end')
        for x in labTkOptions[labTkVar.get()]:
            labxTkVarFilter['menu'].add_command(label=x, command=tk._setit(labxTkVar, x))
    RefreshFrame()


image1TkVar.trace('w', ChangeDropdown)
image2TkVar.trace('w', ChangeDropdown)
labTkVar.trace('w', ChangeDropdown)
labxTkVar.trace('w', ChangeDropdown)
saveasTkVar.trace('w', ChangeDropdown)
RoiCheckTkVar.trace('w', ChangeDropdown)

RefreshFrame()

window.mainloop()
