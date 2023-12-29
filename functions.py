
## Import all packages needed
import pandas as pd

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

def df_corr_subset(normalized_df, corr_df, corr_threshold, label):
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

    # Save the subset dataframe to a CSV file
    subset_filename = f'subset_{label}.csv'
    subset_df.to_csv(subset_filename, index=False)

    return subset_df