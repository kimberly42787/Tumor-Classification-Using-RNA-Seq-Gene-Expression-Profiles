
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


