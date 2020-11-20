# Clean the data

First pipe on the mammography analysis pipeline. Remove some
noise from the data.

Usage is really simple, check it out. 

Author: [gpspelle](https://github.com/gpspelle)  
Date: 18/11/2020

## Clean dataset

The clean dataset is in fact smaller than the original dataset because now
its images are mostly black, because all the noisy background is removed.
Therefore, you can find it here the clean version thus there's no need
to run the code to produce it anymore.

```
unzip clean-mias-dataset.zip
```

## [Optional] Get the original dataset

Since the dataset is pretty small, it was added as a zip file to this repo.
You can extract it using:

``` 
unzip mias-dataset.zip
```

## [Optional] Usage 

Just run the main script:

```
clean_data.py
```

The data paths are hardcoded inside this script, go check it
if you need to change. The script will check if there exists a folder
but will not check the content if the folder that was supposed to exist
exists but contains nothing, for example.

## Requirements

Mainly OpenCV, skimage and numpy.
