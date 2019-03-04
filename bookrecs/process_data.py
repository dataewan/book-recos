import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(filename="data/goodbooks/ratings.csv", test_size=0.2):
    """read the csv datasets.
    Args:
        filename (string): filename for the input csv file.
        test_size (float): proportion of the data to use for test dataset
    Returns: full dataset, train, test dataframes

    """
    dataset = pd.read_csv(filename)
    train, test = train_test_split(dataset) 
    return dataset, train, test


def get_data_summary(dataset):
    """Extract some parameters about the shape of the dataset.

    Args:
        dataset (dataframe): ratings dataset

    Returns: number users, number books

    """
    n_users = len(dataset.user_id.unique())
    n_books = len(dataset.book_id.unique())

    return n_users, n_books
