Sources:

Split same proportion of classes - https://medium.com/@aymuosmukherjee/why-do-we-use-stratify-in-train-test-split-e3eb296a5494
Why normalization - https://discuss.pytorch.org/t/should-we-use-our-normalization-for-training-a-pretrained-model/34905
ImageNet mean and STD - https://www.geeksforgeeks.org/python/how-to-normalize-images-in-pytorch/
To generate only image files - https://matplotlib.org/stable/users/explain/figure/backends.html
Custom Dataset - https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
Freezing and Unfreezing Layers - https://medium.com/we-talk-data/guide-to-freezing-layers-in-pytorch-best-practices-and-practical-examples-8e644e7a9598
Loading and Saving Chechpoints - https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
Guassian Blur - https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
Adding Brightness to image - https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
How to delete a Folder - https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder

Why EfficientNet-B0?
--------------------
1. ~5.3M parameters  - small enough for a few-hundred-image dataset
2. Faster inference than ResNet-50 (~82M FLOPs vs ~4B)