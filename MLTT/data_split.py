from sklearn.model_selection import train_test_split
from typing import TypeVar

T = TypeVar('T')
V = TypeVar('V')

def split_data(X: T, 
               Y: V, 
               train_size: float = 0.5, 
               val_size: float = 0.25, 
               test_size: float = 0.25, 
               random_state: int | None = None, 
               shuffle: bool = False) -> tuple[T, T, T, V, V, V]:
    """
    Splits X and Y dataframes into train, validation, and test sets.

    Parameters:
        X (pd.DataFrame): Features dataframe.
        Y (pd.DataFrame or pd.Series): Target dataframe/series.
        train_size (float): Proportion of the dataset to include in the train set (default 0.7).
        val_size (float): Proportion of the dataset to include in the validation set (default 0.15).
        test_size (float): Proportion of the dataset to include in the test set (default 0.15).
        random_state (int, optional): Random seed for reproducibility.
        shuffle (bool, optional): Whether to shuffle the data before splitting.

    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test: Split dataframes for train, validation, and test sets.
    """
    
    # Ensure that the split sizes add up to 1.0
    assert train_size + val_size + test_size == 1.0, "The sum of train, val, and test sizes must equal 1.0"
    
    # First, split into train and temp (validation + test) sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_size, random_state=random_state, shuffle=shuffle)
    
    # Calculate the relative size of validation and test sets within the temp set
    val_ratio = val_size / (val_size + test_size)
    
    # Split temp into validation and test sets
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, train_size=val_ratio, random_state=random_state, shuffle=shuffle)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
