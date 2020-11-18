from scipy import ndimage as ndi
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

if os.path.isdir(input) and os.path.isdir(mias_input):
    print(" [+] Found that", input, "and", mias_input, "folders exists.")
else:
    print(" [Error] At least one of the input folders doesn't exists.")
    exit(1)


if os.path.isdir(output):
    print(" [+] Found that", output, "folder exists.")
else:
    print(" [!] Found that", output, "folder doesn't exists.")
    print(" [+] Creating the", output, "folder.")
    os.mkdir(output)

total_images = sum(1 for x in truth.iterrows())
total_images_order = len(str(total_images))

for index, row in truth.iterrows():
    start_time = time.time()
    print(" [.] Image #" + str(index+1).zfill(total_images_order), "of #" + str(total_images), end='')
    filename = row['image']

    filepath = os.path.join(input, filename + ".pgm")

    image = cv.imread(filepath, 0)
    ret, thresh1 = cv.threshold(image, 18, 255, cv.THRESH_BINARY)
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
    mask_inv = cv.bitwise_not(mask)

    img = np.zeros(image.shape, np.uint8)
    # Take only region of logo from logo image.
    cv.bitwise_and(original_image, original_image, img, mask = mask)

    output_path = os.path.join(output, filename + ".pgm")
    cv.imwrite(output_path, img)

    print(" - this cycle in seconds:", str(time.time() - start_time).zfill(5), "s", end='\r') 
