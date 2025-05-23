
Just_Handle_door - v1 2023-11-17 8:00am
==============================

This dataset was exported via roboflow.com on January 24, 2024 at 7:35 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3739 images.
Handle-door are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 2 versions of each source image:
* Randomly crop between 0 and 40 percent of the image
* Random rotation of between -23 and +23 degrees
* Random shear of between -22° to +22° horizontally and -22° to +22° vertically
* Random Gaussian blur of between 0 and 4.5 pixels


