<table><tr>
<td><img src="notebook_images/NORMAL2-IM-1423-0001.jpeg" style="width:300px;height:300px"/></td>
<td><img src= "notebook_images/person1945_bacteria_4872.jpeg" style="width:300px;height:300px"/></td>
<td><img src= "notebook_images/person36_bacteria_185.jpeg" style="width:300px;height:300px"/></td> 
</tr></table>

# Diagnosing Pneumonia Using CNNs
**Author:** Evan Holder
Flatiron School - Phase 4 Project

## Repo Contents
* README.MD
* Pneumonia_Classifier.ipynb
* Pneumonia_Classifier-Jupyter-Notebook.pdf
* Diagnosing-Pneumonia-CNN.pdf
* Presentation Link: https://drive.google.com/file/d/1gBzbuWmz_wfz1uRq4lgYccAxKlsjbMGd/view?usp=sharing

## Data
The data are xray images from Mendeley Data. I downloaded the entire dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Introduction and Overview

Pneumonia is a lung infection that affects people of all ages with varying degrees of severity. In fact, the [CDC](https://www.cdc.gov/dotw/pneumonia/index.html) reports that in 2018 about 1.5 million people were diagnosed with pneumonia and more than 40,000 of those people died. For children, pneumonia is the most infectious cause of death worldwide. In 2019, 740,180 children under the age of 5 died of pneumonia according to the [WHO](https://www.who.int/news-room/fact-sheets/detail/pneumonia).  While pneumonia is prevalent and deadly in populations across the world, its also treatable with the proper diagnosis.  For this reason, it's important that the infection is correctly identified so that it can be treated with low-cost and low-tech medication.  

## Business Problem

While pneumonia is diagnosable with symptoms and clinical features, chest x-rays remain the ["gold standard"](https://pneumonia.biomedcentral.com/articles/10.15172/pneu.2014.5/464) for confirming diagnosis of the infection.  Not only are x-rays widely available, but they're also relatively inexpensive and can be reviewed remotely.  With the advent of machine learning, it is possible to further decrease the cost of pneumonia diagnosis through identification of a pneumonia infection by way of convolutional neural networks.  Hospitals or general practices could administer X-rays and subsequently feed the images into a trained CNN to identify pneumonia without the need for a trained physician and/or simply use it to support a diagnosis. Once in use, this CNN would help lower the cost for both healthcare providers and patients themselves.

## Data Understanding

In this section are all import statements. I'll also looked at the layout of the dataset.  The dataset was already split into training/validation/hold-out sets but disproportionately balanced (89-11-.3%). I moved images between sets to balance the datasets at 70-15-15% (train-validation-test).  Additionally I looked at the class balance in each set (training - 81% pneumonia/19% normal, validation 59% pneumonia/ 41% normal, testing- 50/50).

Next I displayed an image and it's size.  Our image data are all in grayscale and at a high resolution.  These are important notes to make as it will affect the setup of our CNN.

In this section I also defined some helper functions to set up for modeling. Specifically I defined functions to visualize training results, log evaluation metrics in a dataframe, and a function to actually train each model.

## Modeling

This section will be dedicated to modeling. Each new model will be iteratively named M_0, M_1,... M_n. Models will vary:
* architecture
* optimizers
* number of epochs trained for
* learning rates
* regularization
* image augementation techniques
* image enhancement with histogram equalization


## Evaluation

This section focuses on picking the best model. Models ranged in performance from 59-91% accuracy. Choosing a model based on accuracy alone is not the best choice. As seen while training, most of these models had sizable variability in accuracy scores of epochs of training. Inconsistent training indicates that the model is either overfitting on the training data, or randomly classifying images without truely finding meaningful patterns. Either way, a model with such drastic fluctuations is not likely to generalize well to unseen data. For this reason, it's best to choose the model with a consistent, reliable training record. The raw results are below

<table><tr>
<td><img src="https://github.com/EvanHolder/Pneumonia-Classifier/blob/main/notebook_images/final_scores.PNG?raw=true" style="width:350px;height:125px"/></td>
<td><img src="https://github.com/EvanHolder/Pneumonia-Classifier/blob/main/notebook_images/final_cm.PNG?raw=true" style="width:350px;height:275px"/></td>
</tr></table>

The final model was tested on test set and received an overall 88% accuracy, even better than the validation data. False negatives did increase from the model's test by 2% from the validation set. The model was 95% precise with pneumonia predictions (correctly classified pneumonia with 95% accuracy) and 83% precise with normal predictions.

<p align="center">
<img src= "https://github.com/EvanHolder/Pneumonia-Classifier/blob/main/notebook_images/model_test_results.PNG?raw=true" style="width:600px;height:391px" class="center"/>
</p>
The above barplot shows the accuracies achieved by the final model when tested on each set of data.  The model achieved a greatest accuracy of 89% on the training set as expected.  The model performed at 86% accuracy on the validation set. The model actually performed better on the testing set than on the validation set which is an indication that the model did generalize well with a 88% accuracy.  Also important to note that the validation set and the testing set are the same size (898 samples) and therefore equally reliable.

<p align="center">
<img src= "https://github.com/EvanHolder/Pneumonia-Classifier/blob/main/notebook_images/sensitivity_specificity_thresholds.PNG?raw=true" style="width:500px;height:302px" class="center"/>
</p>
And finally, I took a look at the true positive and true negative rates as compared with a range of decision thresholds. It was concluded that the tradeoff in lowering the decision threshold to 0.42 to obtain more true positives is worth the decrease in true negatives. Below shows the confusion matrix when using a decision threshold of 0.42, which increased the model's overall accuracy from 88% to 89%.

<table><tr>
<td><img src="https://github.com/EvanHolder/Pneumonia-Classifier/blob/main/notebook_images/final_scores_decrease_t.PNG?raw=true" style="width:350px;height:123px"/></td>
<td><img src="https://github.com/EvanHolder/Pneumonia-Classifier/blob/main/notebook_images/final_cm_decrease_t.PNG?raw=true" style="width:350px;height:263px"/></td>
</tr></table>


## Next Steps 
This model could improve by reducing the 67 false negatives. These are people who will not receive a pneumonia diagnosis and consequently treatment for their infection due to their missclassification by the model.  The model will need to be further improved with either more validation, architectural changes, more images to train on, or all of the above. Another way to improve the model would be have trained medical profession view the output of the feature maps from the model.  

<p align="center">
<img src= "https://github.com/EvanHolder/Pneumonia-Classifier/blob/main/notebook_images/feature_maps.PNG?raw=true" style="width:800px;height:194px" class="center"/>
</p>
Above, I have displayed a few of the feature maps from the first convolutional layer of the NN.  A trained eye could review this patterns highlighted by the model and tune the model according to more closely follow pneumonia infections as represented on xrays.