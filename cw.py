import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt


LUTLog = {}
tmp = 255/np.log(256)
for x in range(256):
    LUTLog[x] = np.log(1+x)*tmp


LUTPower = {}
gamma = .01
tmp = 255**(1 - gamma)
for x in range(256):
    LUTPower[x] = tmp*x**gamma


RandLUT = {}
for x in range(256):
    RandLUT[x] = np.random.randint(256)

bit = 4

def ClippingImage(imgTmp, norm=False):
    imgTmp = imgTmp.astype(int)
    if norm:
        imgTmp = ((imgTmp - np.min(imgTmp))/np.max(imgTmp))*255
    imgTmp[imgTmp < 0] = 0
    imgTmp[imgTmp > 255] = 255
    return imgTmp


def ReadImage(path):
    if path.endswith('.raw'):
        imgTmp = np.fromfile(path, dtype='int8', sep="")
        s = int(math.sqrt(imgTmp.shape[0]))
        imgTmp = imgTmp.reshape(s,s)
        original = imgTmp
    else:
        original = cv.imread(path)
        imgTmp = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])
    imgTmp = ClippingImage(imgTmp)
    return original, imgTmp


def ShowImages(original, processed):
    if isinstance(original, list) == False:
        original = [original]
    if isinstance(processed, list) == False:
        processed = [processed]
    l = len(original)
    i = 1
    for x in list(zip(original, processed)):
        plt.subplot(l, 2, i),plt.imshow(x[0], cmap='gray'),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(l, 2, i+1),plt.imshow(x[1], cmap='gray'),plt.title('Processed')
        plt.xticks([]), plt.yticks([])
        i += 2
    plt.show()


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
        res = 256 - 1 - imgTmp
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
    res = ClippingImage(res, True)
    return res


def Histogram(imgTmp, norm=False):
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
    dst = (cv.filter2D(np.float32(img), -1, kernel)).astype(int)
    return dst




# path = 'images/Cameraman.tif'
# path = 'images/Goldhill.tif'
path = 'images/Baboon.bmp'
# path = 'images/Peppers.raw'

original, img = ReadImage(path)

# lab 2
# processed = RescaleImage(img, 1.2)
# processed = ShiftingImage(img)
# processed = ShiftingImage(img, 'all_random')



#lab 3
path2 = 'images/Baboon.bmp'
original2, img2 = ReadImage(path2)

roi = np.random.choice([0,1], img2.shape)
processed = ArithmeticOperations(img2, roi, 'multiplication')
processed = BinaryNot(img2)
processed = BinaryOperations(img, img2, 'xor')
processed2 = BinaryOperations(img2, img, 'xor')



# lab 4

processed = ImageTransformation(img, 'logarithmic')
processed2 = ImageTransformation(img, 'bit_plane_slicing')
# processed3 = ImageTransformation(img, 'random_look_up_table')

ShowImages([img, img], [processed, processed2])



#  lab 5


hist = Histogram(img)
plt.bar(list(range(1,257)), hist)
plt.show()

hist = Histogram(img, True)
plt.bar(list(range(1,257)), hist)
plt.show()


imgCol = cv.cvtColor(original2, cv.COLOR_BGR2YUV)
imgCol[:,:,0] = cv.equalizeHist(imgCol[:,:,0])
imgCol = cv.cvtColor(imgCol, cv.COLOR_YUV2BGR)

ShowImages(original2, imgCol)



# lab 6

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

# dst = Convolution(img, kernel["averaging"])
# dst = Convolution(img, kernel["weighted_averaging"])
# dst = Convolution(img, kernel["4_neighbour_laplacian"])
# dst = Convolution(img, kernel["8_neighbour_laplacian"])
# dst = Convolution(img, kernel["4_neighbour_laplacian_enhancement"])
# dst = Convolution(img, kernel["8_neighbour_laplacian_enhancement"])
# dst = Convolution(img, kernel["roberts"])
# dst = Convolution(img, kernel["roberts_with_absolute_value"])
# dst = Convolution(img, kernel["sobel_x"])
# dst = Convolution(img, kernel["sobel_y"])
# dst = Convolution(img, kernel["gaussian"])
dst = Convolution(img, kernel["laplacian_of_gaussian"])

ShowImages(img, dst)



#  lab 7


def SubMatrices(mat):
    sub_shape = (3, 3)
    view_shape = tuple(np.subtract(mat.shape, sub_shape) + 1) + sub_shape
    strides = mat.strides + mat.strides
    sub_matrices = np.lib.stride_tricks.as_strided(mat, view_shape, strides)
    return sub_matrices


midPoint = lambda x: (min(x)+max(x))/2


noise = np.random.choice([0,255], img.shape)
imgTmp = ClippingImage(img + noise)


sub = SubMatrices(imgTmp)
shape = sub.shape
totDim = shape[0] * shape[1]
sub = sub.reshape(1, totDim, 9)
# sub = np.apply_along_axis(np.min, 2, sub)
sub = np.apply_along_axis(midPoint, 2, sub)
sub = ClippingImage(sub.reshape(shape[0], shape[1]))

ShowImages(imgTmp, sub)



#  lab 8

def Threshold(imgTmp2, limit=[]):
    imgTmp = np.array(imgTmp2, copy=True)
    if len(limit) == 2:
        imgTmp[imgTmp < limit[0]] = 0
        imgTmp[imgTmp > limit[1]] = 0
    elif len(limit) == 1:
        imgTmp[imgTmp < limit[0]] = 0
    imgTmp[imgTmp > 0] = 255
    return imgTmp


hist = Histogram(img)
mean = int(np.mean(img))
std = int(np.std(img))

plt.bar(list(range(1,257)), hist)
plt.show()

thresholded = Threshold(img, [mean])
ShowImages(img, thresholded)




tmin = mean - std
tmax = mean + std
thresholded = Threshold(img, [tmin, tmax])

ShowImages(img, thresholded)


thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

ShowImages(img, thresholded)




path = 'images/Baboon.bmp'
original, img = ReadImage(path)





