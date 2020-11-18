# Clean the data

First pipe on the mammography analysis pipeline. Remove some
noise from the data.

Usage is really simple, check it out. 

Author: [gpspelle](https://github.com/gpspelle)
Date: 18/11/2020

## Get the dataset

Since the dataset is pretty small, it was added as a zip file to this repo.
You can extract it using:

``` 
unzip mias-dataset
```

## Usage 

Just run the main script:

```
clean\_data.py
```

The data paths are hardcoded inside this script, go check it
if you need to change. The script will check if there exists a folder
but will not check the content if the folder that was supposed to exist
exists but contains nothing, for example.

## Requirements

Mainly OpenCV, skimage and numpy.
