from scipy import ndimage as ndi
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte

import cv2 as cv
import pandas as pd

import os
import time

import psutil
from multiprocessing import Pool

num_cpus = psutil.cpu_count(logical=False)

def calculate_gaussian_kernel(size=3, sigma=1):
    kernel = np.zeros((size, size))
    for g_i, k_i in zip(range(int(-size/2), int(size/2) + 1), range(0, size)):
        for g_j, k_j in zip(range(int(-size/2), int(size/2)+1), range(0, size)):
            kernel[k_i][k_j] = np.exp(-(g_i*g_i + g_j*g_j)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)

    return kernel


def check_create(folder):
    if os.path.isdir(folder):
        print(" [+] Found that", folder, "folder exists.")
    else:
        print(" [!] Found that", folder, "folder doesn't exists.")
        print(" [+] Creating the", folder, "folder.")
        os.mkdir(folder)

truth_file = "truth.csv"
if os.path.isfile(truth_file):
    print(" [+] Found that", truth_file, "exists.")
else:
    print(" [Error] You need the", truth_file, "file.")
    print(" [Error] This file is a csv file with the expected output for the data.")
    exit(1)

truth = pd.read_csv(truth_file)

mias_input = "mias-dataset/data"
input = "mias-dataset/data"
output = "clean-mias-dataset"
output_image = os.path.join(output, "data")
output_mask = os.path.join(output, "mask")

total_images = sum(1 for x in truth.iterrows())
total_images_order = len(str(total_images))

gaussian_kernel = calculate_gaussian_kernel(size=30, sigma=5)

def clean_image(data):

    index, row = data
    start_time = time.time()
    filename = row['image']

    filepath = os.path.join(input, filename + ".pgm")

    image = cv.imread(filepath, 0)
    _, thresh1 = cv.threshold(image, 18, 255, cv.THRESH_BINARY)
    image = thresh1

    filepath = os.path.join(mias_input, filename + ".pgm") 
    original_image = cv.imread(filepath, 0)

    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, markers)

    unique_entries = np.unique(labels)

    contours = []

    for u in unique_entries:
        index_labels = np.where(labels == u)
        contours.append(np.array([[j, i] for i, j in zip(index_labels[0], index_labels[1])]).reshape((-1,1,2)).astype(np.int32))

    contours_areas = []
    contours_colors = []
    for c in contours:
        area = len(c)
        mask = np.zeros(image.shape, np.uint8)

        cv.drawContours(mask, c, -1, 255, -1)

        mean = cv.mean(original_image, mask=mask)[0]

        if mean > 80:
            contours_colors.append(mean)
            contours_areas.append(area)
        else:
            contours_colors.append(0)
            contours_areas.append(0)


    biggest = np.argmax(contours_areas)
    mask = np.zeros(image.shape, np.uint8)
    mask = cv.drawContours(mask, contours[biggest], -1, 255, -1)

    gaussian_mask = signal.convolve2d(mask, gaussian_kernel, boundary='symm', mode='same')

    _, gaussian_mask = cv.threshold(gaussian_mask, 255//2, 255, cv.THRESH_BINARY)
    gaussian_mask = gaussian_mask.astype('uint8')

    img = np.zeros(image.shape, np.uint8)

    # Take only region of interest from the mammogram image.
    cv.bitwise_and(original_image, original_image, img, mask=gaussian_mask)

    # Save the output image
    output_image_path = os.path.join(output_image, filename + ".pgm")
    cv.imwrite(output_image_path, img)

    # Save the mask that produced the image
    output_mask_path = os.path.join(output_mask, filename + ".pgm")
    cv.imwrite(output_mask_path, gaussian_mask)

    print(" [.] Image #" + str(index+1).zfill(total_images_order), "of #" + str(total_images), "- this cycle in seconds:", str(time.time() - start_time).zfill(5), "s")

if __name__ == '__main__':
    if os.path.isdir(input) and os.path.isdir(mias_input):
        print(" [+] Found that", input, "and", mias_input, "folders exists.")
    else:
        print(" [Error] At least one of the input folders doesn't exists.")
        exit(1)

    check_create(output)
    check_create(output_image)
    check_create(output_mask)

    # Parallel run
    with Pool(num_cpus) as p:
        p.map(clean_image, truth.iterrows())

    # Serial run
    #for data in truth.iterrows():
    #    clean_image(data)

