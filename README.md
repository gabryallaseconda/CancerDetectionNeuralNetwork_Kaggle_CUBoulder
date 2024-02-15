# Histopathologic Cancer Detection using Convolutional Neural Networks with PyTorch 

Kaggle Playground Competition ([Link](https://www.kaggle.com/competitions/histopathologic-cancer-detection))

Project for Deep Learning at CU Boulder

## Project Notes

My solution is based on Convolutional Neural Networks. The final model was selected through cross-validation. It is built upon a pre-trained ResNet34; larger sizes take much longer to train, while smaller ones yield poor results. I chose to balance the classes through augmentation: negative classes are augmented by a factor of 15, while positive ones by a factor of 10 (see `DataGenerator` in `tools.py`).

## Notebooks and Files Description

- *exploration.ipynb*: EDA, used only to check class balance.
- *crossvalidation.ipynb*: Code to run cross-validation. Note that the entire process involves only a subset of the given data.
- *train.ipynb*: Code to train the final model. Training is performed using the full train dataset (as provided).
- *inference.ipynb*: Code to make predictions using the chosen model.
- *tools.py*: All the functions used.

## Relevant Articles

- Scientific paper on exactly the same topic, using neural networks: [Link](https://arxiv.org/pdf/2311.07711.pdf)
- Article on preprocessing for this competition: [Link](https://towardsdatascience.com/data-preparation-guide-for-detecting-histopathologic-cancer-detection-7b96d6a12004)
- Article from one of the best submission authors: [Link](https://sergeykolchenko.medium.com/histopathologic-cancer-detection-as-image-classification-using-pytorch-557aab058449)
- Code for the previous article: [Link](https://github.com/azkalot1/Histopathologic-Cancer-Detection) Note: Some functions have been taken from this repository; this code is quite old, and there are many outdated functions that do not work with current package versions.
- Article from another one of the best submission authors: [Link](https://www.kaggle.com/competitions/histopathologic-cancer-detection/discussion/87397)
- Article about test-time augmentation: [Link](https://medium.com/analytics-vidhya/test-time-augmentation-using-pytorch-3da02d0a3188)
- Article about test-time augmentation: [Link](https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d)
