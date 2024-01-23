# Project Title: Tumor Classification Using RNA-Seq Gene Expression Profiles

## Project Description
The goal of this project is to analyze RNA-seq data taken from various cancer patients and implement classification models for the accurate indentification of tumor types. The project consists of comprehensive data analysis, exploration, pre-processing, and the application of classification algorithms. 

The data used for this project can be found here: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

## **Table of Contents**
  1. [Data Exploration](#data_exploration)
  2. [Data Pre-Processing](#data_preprocessing)
  3. [Classification Models](#classification_models)

<a name="data_exploration"></a>
### Data Exploration

The first step of my pipeline is to perform the EDA. This will give us a better understanding of the dataset we are working with. 

After merging the data file and the labels file, the resulting shape of the dataset is (801, 20533). Whether this dataset is deemed as "big data" is subjective and contingent on its intended use. In the context of gene expression data, the sheer volume of genes present in the human body, many of which may not be relevant to the specific case at hand, diminishes the perception of this dataset as big data. In reality, this amount of gene expression data represents only a fraction of what researchers typically encounter when profiling various diseases.

Once I gathered information on the dataset, the next step involved data exploration and cleanup. Here are the key aspects of the data exploration conducted:

**1. Checking and Handling Null Values:**

Fortunately, the dataset did not contain any null values.

**2. Handling Columns with Zero Values:**

In addition to checking for null values, an examination was conducted to identify columns with all zero values. While this step may not be necessary for every dataset, it was deemed essential in this context. The objective was to check if there were any genes in the dataset with zero expression across all samples.

<img width="494" alt="All Zero Values Counts" src="https://github.com/kimberly42787/RNA-Seq_GeneExpression_Model/assets/56846342/6fd862e7-13ea-448f-81d1-9a9879b54d94">

The visualization above illustrates the count of columns (genes) with all zero values and those with atleast one non-zero value, providing insights into the distribution of such genes in the dataset.

Knowing that those genes with all zero values will not effect our model, we can clean up our dataset a little more by removing all the columns (genes) with all zero values. Our columns count went from 20533 genes to 20266 genes.

**3. Dealing with Imbalanced Data:**

Next, I checked class distribution. The dataset have 5 unique tumor types. I want to check the distribution of my data set. From this bar graph, BRCA has the highest distribution at 300 and COAD having the lowest at 78. If the model is trained with this class distribution, then the model will favor certain classes over the other as the data is imbalanced. 

ADD GRAPH


There are different ways to handle imbalanced data for classsification models. Some of the ways are:
- Using specific algorithm that are less sensitive to imbalanced data. Some classification algorithms, such as Random Forest and Gradient Boosting, are more robust in handling imbalanced datasets. 
- Adjusting class weights during model training.
- Resampling: 
  - One way is to undersample the majority class. This involves removinf samples from the majority classes to balance out the minority classes. 
  - Another way is oversampling the minority classes. This would generate synthetic samples for the minority classes to balance out the class distribution. 
  - Another way is SMOTE. This stands for Synthetic Minority Over-Sampling Technique. This technique generates synthetic samples for the minority classes. 

For this dataset and modeling, I will be using the SMOTE technique to handle the imbalanced data. This technique is different from oversampling the minority classes. The SMOTE technique generates synthetic samples for the minority classes by considering the features of the existing minority classes and their nearest neighbors. Unlike oversampling the minority classes, this technique helps to avoid overfitting on duplicated samples. 

ADD GRAPH 

This visualization showed the class distribution after applying SMOTE to balance the dataset. Now, all of our unique tumor types contain 300 datasets. The cleaned, balanced dataset is save as Gen_Expression_Data.csv to be used for our Data Pre-processing

<a name="data_preprocessing"></a>
### Data Pre-Processing

Currently, the dataset is not in the right format to process for a machine learning model. This section, I will go through the steps I did to transform the dataset for my model. 

**1. Encoding Categorical Labels with Numerical Values**

LabelEncoder() from the scikit-learn library was used to turn our categorical columnm ("Class") that holds the tumor type labels. The objective of this is to convert the categorical labels into numerical representations suitable for training classification models. The encoded values are then stored in a new column, "class_encoded" which will be used in the machine learning pipeline later on. 

**2. Data Normalization using MinMaxScaler()**
Given the inherent variability in gene expression values, it is paramount to normalize the data before incorporating it into our model. Data normalization is a critical step in ensuring that genes with disparate expression magnitudes do not unduly influence the analysis, mitigating the risk of false results. In this project, I opted for the MinMaxScaler class from scikit-learn to achieve this normalization. The MinMaxScaler scales the values of each gene's expression to a specified range, typically [0, 1]. This uniform scaling is essential to prevent genes with larger expression values from disproportionately impacting the analysis. The resultant normalized gene expression data, with values confined to a consistent scale, serves as the input for both training and testing our model, contributing to the model's robustness and interpretability.

**3. Subset Data for Testing**

In order to assess the performance of our machine learning model, a dedicated subset of the data has been created for testing purposes. This subset is distinct from the training and testing datasets and is designed to evaluate the model's generalization capabilities on unseen data. By employing a separate subset for testing, we aim to obtain a more accurate representation of the model's real-world performance. The subset data has been carefully curated to encompass a diverse range of samples, ensuring robust evaluation metrics that reflect the model's efficacy beyond the training set.

**4. Separation of Features and Target Value**
To prepare the data for model training, I separated the dataset into two subsets, features and target value. The features represent the gene expression data that will used as the input variables used for prediction. The target value is the ouput variable we aim to predict which is the tumor types. 

**5. Data Splitting into Training and Testing Sets**

The data is then split into the training and testing sets using the train_test_split() function from scikit-learn. 80% of the data will be used as the training set, while the remaining 20% is for the testing set. The random_state parameter is set to 42 to ensure reproducibility of the split. This approach ensures a randomized dataset feeding into the moddel, so it can learn from a diverse set of samples that contributes to a more reliable assessment of the model. 

**6. Feature Selection** 
The last step of Data Pre-processing is feature selection. PCA (Principal Component Analysis) has been applied for feature selection. PCA is dimensionality reduction technique that transform high-dimensional data into a lower dimensional representation while retaining as much of the original variability as possible. The transformed featirs will be used as the input for the model. 

<a name="classification_models"></a>
### Classification Models

