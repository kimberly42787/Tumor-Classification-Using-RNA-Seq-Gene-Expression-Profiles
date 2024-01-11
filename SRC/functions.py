
## Import all packages needed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Model Evaluation
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize
import config

from sklearn.feature_selection import mutual_info_classif
import numpy as np

# Data Visualization 
import seaborn as sns
import matplotlib.pyplot as plt


### Data Preprocessing Functions

## Print columns that has null values
def print_columns_with_null(df):

    """ 
    This function allows us to go through each ~20,000 columns to see if any
    contains null values and prints out which columns contains the null values
    and how many. If not, it will simply say " No columns have null values"

    """

    # Calculate the number of null values in each column
    null_counts = df.isnull().sum()

    # Filter columns with null values
    columns_with_null = null_counts[null_counts > 0]

    # Print the result
    if not columns_with_null.empty:
        print("Columns with null values:")
        for column, null_count in columns_with_null.items():
            print(f"{column}: {null_count} null values")
    else:
        print("No columns have null values.")

## Print out the count of columns with all zero values and atleast one non-zero values
def count_columns_with_zeros(df):

    """
    This functon will allow us to ireterate through all of ~20,000 columns 
    and count how many contain all zero values and how many contains atleast 
    one non-zero values. Then, it will create a dataframe with the counts so
    it will be easy to visualize.
    """
    
    # Initialize counters
    all_zeros_count = 0
    non_zero_count = 0

    # Iterate through each column
    for column in df.columns:
        # Check if all values in the column are 0
        if (df[column] == 0).all():
            all_zeros_count += 1
        else:
            non_zero_count += 1
    
    # Create a DataFrame with the counts
    result_df = pd.DataFrame({
        'Category': ['All Zeros', 'Non-Zero'],
        'Count': [all_zeros_count, non_zero_count]
    })

    return result_df 

## Remove columns with all zero values from our dataframe
def remove_cols_with_all_zero(df):

    """ 
    This function will identify the columns with all zero values
    and remove it from our dataframe.
    
    I could just use this function without using the 
    "count_columns_with_zeros" function to make it faster. For this purpose, I 
    wanted to visualize the comparison.
    """

    #Identify the columns with all zero values
    all_zero_cols = df.columns[(df == 0).all()]

    #Remove the columns from the dataframe
    filtered_df = df.drop(columns=all_zero_cols)

    #Return our filtered df
    return filtered_df 

## Create a subset dataframe from an original dataframe
def df_corr_subset(normalized_df, corr_df, corr_threshold):
    
    """ 
    This function will allow us to create a subset dataframe from 
    our original normalized dataframe that contains the columns that 
    has a certain correlation to our target variable. These subsets 
    will be used to test our models
    """
    
    # Filter genes based on correlation threshold
    selected_genes = corr_df[corr_df['Correlation'] >= corr_threshold]['Gene']

    # Create a subset of the original normalized dataframe
    subset_df = normalized_df[['Unnamed: 0', 'Class'] + list(selected_genes)]

    return subset_df

# Create our dictionary into dataframe function
def dict_to_df(data_dict, column_names):
    """
    This function converts a dictionary into a dataframe

    Variable needed: 
        - Column names (column_names)
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')

    # Reset the index to create a column with the original dictionary keys
    df.reset_index(inplace=True)
    
    # Rename the columns based on the provided column names
    df.columns = column_names
    
    return df

# Save dataframe into a CSV
def save_dataframe_to_csv(df, subfolder_path, csv_filename):
    """
    Save a DataFrame to a CSV file in a specified subfolder.

    Parameters:
    - df: pandas DataFrame
    - subfolder_path: str, relative path to the subfolder within the VS Code project
    - csv_filename: str, name of the CSV file

    Returns:
    - None
    """
    # Ensure the subfolder exists, create it if necessary
    os.makedirs(subfolder_path, exist_ok=True)

    # Save the DataFrame to a CSV file in the specified subfolder
    df.to_csv(os.path.join(subfolder_path, csv_filename), index=False)



### Classification Model Functions

# Write a function to split our training and testing data set
def split_train_test_data(X, Y):
    """
    Function splits our features and target variable from each other.
    
    The data is then divided into our "training" set with 70% of our data and 
    our "rest" set with 30% of our data for our first initial split. 
    Then, we split our "rest" data 50% so we have 15% for our "validation" and 
    15% for our test
    """

    # Our features all start fron the 4th column and our target is "Class"
    X = X # features
    Y = Y # target variable 

    # Split our data into training, validation, and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42) # 80% is for training

    return x_train, x_test, y_train, y_test

# Hyperparameter_grid_search 
def hyperparameter_grid_search(x_val, y_val, param_grid):


    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state = None),
                               param_grid = param_grid, 
                               cv = 5)
    grid_search.fit(x_val, y_val)

    # Get the best parameter for the subset
    best_params = grid_search.best_params_

    return best_params

# Model Evaluation 
def model_evaluate(y_test, y_pred, labels, pred_prob):
    # Accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Precision
    precision = precision_score(y_test, y_pred, average = 'weighted')
    print(f"Precision: {precision:.4f}")

    # recall
    recall = recall_score(y_test, y_pred, average = 'weighted')
    print(f"Recall: {recall:.4f}")

    # f1 score
    f1score = f1_score(y_test, y_pred, average = 'weighted')
    print(f"f1 score: {f1score:.4f}")

    # Class Report
    class_report = classification_report(y_test, y_pred, target_names = labels)
    print('\n')
    print('Classification Report\n')
    print(class_report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns = labels)
    print(cm_df)

    # Binarize the labels for each class
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Get decision function scores for each class
    # y_score = ovr_rf_model.decision_function(X_test_selected)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = len(np.unique(y_test))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure(figsize=(7, 5))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each class')
    plt.legend(loc='lower right')
    plt.show()








    