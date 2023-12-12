#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
from scipy.signal import gaussian
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import timeit
#sys.path.append("../../p1/code") # set the path for visualPercepUtils.py
import visualPercepUtilsMod01 as vpu

# -----------------------
# Task 2 - 3 - a 
# -----------------------

n = 50  # Пример размера массива Array size example
sigma = 5  # Пример стандартного отклонения Standard Deviation Example

# Генерация 1D Гауссова массива Generating a 1D Gaussian Array
gv1d = gaussian(n, sigma)


#Visualization Using Matplotlib

# Визуализация 1D Гауссова массива # Visualization of a 1D Gaussian array
plt.imshow([gv1d], aspect='auto', interpolation='none', cmap='gray')
plt.colorbar()  # Добавляем цветовую шкалу для наглядности # Add a color scale for clarity
plt.title("1D Gaussian Array")
plt.show()

# -----------------------
# Task 2 - 3 - b 
# -----------------------

#Aussian distribution, we can use the one-dimensional Gaussian 
#vector gv1d that has already been generated and obtain the two-dimensional
# matrix gv2d by outer product of the vector gv1d with its transposed version. 
# This can be done using the np.outer function from NumPy.
gv2d = np.outer(gv1d, gv1d)

# Визуализация 2D Гауссова распределения
plt.imshow(gv2d, interpolation='none', cmap='viridis')  # Используем цветовую карту для наглядности
plt.colorbar()  # Добавляем цветовую шкалу
plt.title("2D Gaussian Distribution")
plt.show()


def quotientImage(im, sigma):
    # Применяем Гауссовское размытие к изображению Applying Gaussian Blur to an Image
    blurred_im = gaussian_filter(im, sigma=sigma)

    # Вычисляем квотиентное изображение Calculating the quota image
    # Для избежания деления на ноль добавляем небольшое значение к знаменателю
    # To avoid division by zero, add a small value to the denominator
    epsilon = 1e-10
    quotient_im = im / (blurred_im + epsilon)

    return quotient_im

# Тестирование функции testing
image_path = './imgs-P2/3.png'  
image_path1 = './imgs-P2/21.png' 
original_im = np.array(Image.open(image_path).convert('L'))  # Convert to gray image
original_im1 = np.array(Image.open(image_path1).convert('L'))  # Convert to gray image

sigma = 10  # Пример стандартного отклонения для Гауссовского размытия Example of standard deviation for Gaussian blur
quotient_im = quotientImage(original_im, sigma)
quotient_im1 = quotientImage(original_im1, sigma)

# Выводим оригинальное и квотиентное изображения We display the original and quota images
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(original_im, cmap='gray')
plt.title('Original image 1')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(quotient_im, cmap='gray')
plt.title('Quotent image 1')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(original_im1, cmap='gray')
plt.title('Original image 2')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(quotient_im1, cmap='gray')
plt.title('Quotent image 2')
plt.axis('off')

plt.show()










# -----------------------
# Salt & pepper noise
# -----------------------

def addSPNoise(im, percent):
    # Now, im is a PIL image (not a NumPy array)
    # percent is in range 0-100 (%)

    # convert image it to numpy 2D array and flatten it
    im_np = np.array(im)
    im_shape = im_np.shape  # keep shape for later use (*)
    im_vec = im_np.flatten()  # this is a 1D array # https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/

    # generate random locations
    N = im_vec.shape[0]  # number of pixels
    m = int(math.floor(percent * N / 100.0)) # number of pixels corresponding to the given percentage
    locs = np.random.randint(0, N, m)  # generate m random positions in the 1D array (index 0 to N-1)

    # generate m random S/P values (salt and pepper in the same proportion)
    s_or_p = np.random.randint(0, 2, m)  # 2 random values (0=salt and 1=pepper)

    # set the S/P values in the random locations
    im_vec[locs] = 255 * s_or_p  # values after the multiplication will be either 0 or 255

    # turn the 1D array into the original 2D image
    im2 = im_vec.reshape(im_shape) # (*) here is where we use the shape that we saved earlier

    # convert Numpy array im2 back to a PIL Image and return it
    return Image.fromarray(im2)


from scipy.ndimage import convolve1d

def averageFilterSep(im, filterSize):
    # Создаем одномерный фильтр
    one_d_filter = np.ones(filterSize) / filterSize
    
    # Применяем фильтр к строкам, затем к столбцам
    filtered_rows = convolve1d(im, one_d_filter, axis=0)
    filtered_image = convolve1d(filtered_rows, one_d_filter, axis=1)

    return filtered_image

def testSandPNoise(im, percents):
    imgs = []
    for percent in percents:
        imgs.append(addSPNoise(im, percent))
    return imgs


# -----------------
# Gaussian noise
# -----------------

def addGaussianNoise(im, sd=5):
    if len(im.shape) == 3:  # Проверка, является ли изображение цветным Checking if an image is in color
        noisy_im = np.zeros_like(im)
        for i in range(3):  # Применяем шум к каждому цветовому каналу Apply noise to each color channel
            noisy_im[:, :, i] = im[:, :, i] + np.random.normal(0, sd, im[:, :, i].shape)
    else:  # Изображение в оттенках серого Grayscale Image
        noisy_im = im + np.random.normal(0, sd, im.shape)
    
    # Обрезаем значения, выходящие за пределы допустимого диапазона Trimming values outside the acceptable range
    noisy_im = np.clip(noisy_im, 0, 255)

    return noisy_im

# Загружаем изображение Uploading an image
image_path = './imgs-P2/2.png'  # путь к  изображению path to image
original_im = np.array(Image.open(image_path))

# Применяем Гауссовский шум Applying Gaussian noise
noise_sd = 10  # Пример стандартного отклонения шума Example of standard deviation of noise
noisy_im = addGaussianNoise(original_im, noise_sd)

# Выводим оригинальное и зашумленное изображения We display the original and noisy images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_im, cmap='gray')
plt.title('Original image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_im, cmap='gray')
plt.title(f'Noisy image (sd={noise_sd})')
plt.axis('off')

plt.show()

def testGaussianNoise(im, sigmas):
    imgs = []
    for sigma in sigmas:
        print('testing sigma:', sigma)
        imgs.append(addGaussianNoise(im, sigma))
        print(len(imgs))
    return imgs


# -------------------------
# Average (or mean) filter
# -------------------------

def averageFilter(im, filterSize):
    mask = np.ones((filterSize, filterSize))
    mask = np.divide(mask, np.sum(mask)) # can you think of any alternative for np.sum(mask)?
    return filters.convolve(im, mask)


def testAverageFilter(im_clean, params):
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg) # salt and pepper noise
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(averageFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Gaussian filter
# -----------------

def gaussianFilter(im, sigma=5):
    # im is PIL image
    return filters.gaussian_filter(im, sigma)


def testGaussianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sigma in params['sd_gauss_noise']:
        im_dirty = addGaussianNoise(im_clean, sigma)
        for filterSize in params['sd_gauss_filter']:
            imgs.append(np.array(im_dirty))
            imgs.append(gaussianFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Median filter
# -----------------

def medianFilter(im, filterSize):
    return medfilt2d(im, filterSize)

def testMedianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg)
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(medianFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Test image files
# -----------------

path_input = './imgs-P2/'
path_output = './imgs-out-P2/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'lena512.pgm']  # lena256, lena512

# --------------------
# Tests to perform
# --------------------

testsNoises = ['testSandPNoise', 'testGaussianNoise']
testsFilters = ['testAverageFilter', 'testGaussianFilter', 'testMedianFilter']
bAllTests = True
if bAllTests:
    tests = testsNoises + testsFilters
else:
    tests = ['testSandPNoise']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testGaussianNoise': 'Gaussian noise',
             'testSandPNoise': 'Salt & Pepper noise',
             'testAverageFilter': 'Mean filter',
             'testGaussianFilter': 'Gaussian filter',
             'testMedianFilter': 'Median filter'}

bSaveResultImgs = False

# -----------------------
# Parameters of noises
# -----------------------
percentagesSandP = [3]  # ratio (%) of image pixes affected by salt and pepper noise
gauss_sigmas_noise = [3, 5, 10]  # standard deviation (for the [0,255] range) for Gaussian noise

# -----------------------
# Parameters of filters
# -----------------------

gauss_sigmas_filter = [1.2]  # standard deviation for Gaussian filter
avgFilter_sizes = [3, 7, 15]  # sizes of mean (average) filter
medianFilter_sizes = [3, 7, 15]  # sizes of median filter

testsUsingPIL = ['testSandPNoise']  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)

#measure the execution time of both versions of the filter for different 
#image sizes and masks. We can use the timeit module for this.


def measure_time(im, filter_size, function):
    start_time = timeit.default_timer()
    function(im, filter_size)
    return timeit.default_timer() - start_time

# Тестирование на разных размерах изображений и масок Testing on different sizes of images and masks
image_sizes = [128, 256, 512]  # Примеры размеров изображений Examples of image sizes
filter_sizes = [3, 5, 7]  # Примеры размеров фильтров Examples of filter sizes
times_sep = {size: [] for size in image_sizes}
times_non_sep = {size: [] for size in image_sizes}

for size in image_sizes:
    # Создаем тестовое изображение Create a test image
    test_image = np.random.rand(size, size)

    for filter_size in filter_sizes:
        time_sep = measure_time(test_image, filter_size, averageFilterSep)
        time_non_sep = measure_time(test_image, filter_size, averageFilter)

        times_sep[size].append(time_sep)
        times_non_sep[size].append(time_non_sep)



def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:

            if test == "testGaussianNoise":
                params = gauss_sigmas_noise
                subTitle = r", $\sigma$: " + str(params)
            elif test == "testSandPNoise":
                params = percentagesSandP
                subTitle = ", %: " + str(params)
            elif test == "testAverageFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testMedianFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testGaussianFilter":
                params = {}
                params['sd_gauss_noise'] = gauss_sigmas_noise
                params['sd_gauss_filter'] = gauss_sigmas_filter
                subTitle = r", $\sigma_n$ (noise): " + str(gauss_sigmas_noise) + ", $\sigma_f$ (filter): " + str(gauss_sigmas_filter)
            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
                print("num images", len(outs_np))
            print(len(outs_np))

            # Создание подзаголовков Creating Subheadings
            subtitles = ['Original Image'] + [f'{test} {i+1}' for i in range(len(outs_np))]
           
            # Определение AoI для каждого изображения Determining AoI for each image
            # Например, показываем только центральную часть изображений For example, we show only the central part of the image
            aois = [(50, 50, 150, 150)] * (len(outs_np) + 1)  # Применяем одинаковую AoI ко всем изображениям Applying the same AoI to all images

            # Отображение изображений с подзаголовками и AoI Display images with subtitles and AoI
            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle, subtitles=subtitles, aois=aois)


for size in image_sizes:
    plt.plot(filter_sizes, times_sep[size], label=f'Separable - Size {size}')
    plt.plot(filter_sizes, times_non_sep[size], label=f'Non-separable - Size {size}')

plt.xlabel('Filter Size')
plt.ylabel('Time (seconds)')
plt.title('Filter Performance Comparison')
plt.legend()
plt.show()


if __name__ == "__main__":
    doTests()

