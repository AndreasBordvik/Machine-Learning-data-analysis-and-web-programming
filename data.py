#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def readDataFromCSV(filename: str) -> pd.DataFrame:
    """Function for reading a csv file containing data

    Args:
        filename: File location as string

    Returns:
        data: A pandas dataframe based on input csv
    """
    data = pd.read_csv(filename, index_col=0).dropna()
    return data


def data(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    """The function takes inn a pandas dataframe. Splits the dataframe
    into 80% for training data, and 20% for validation/test data with equal
    proportions of positive samles.

    Args:
        df: Pandas dataframe

    Returns:
        training: A pandas dataframe for training
        validation: A pandas dataframe for validation
    """
    training, validation = train_test_split(df, train_size=0.8,
                                            test_size=0.2, stratify=df.diabetes)
    training.reset_index(inplace=True, drop=True)
    validation.reset_index(inplace=True, drop=True)
    return training, validation


def scatterplot(df: pd.DataFrame, x: str, y: str) -> plt.Figure:
    """Function for plotting x and y features of a pandas dataframe

    Args:
        df: Pandas dataframe
        x: feature to use for x-axis when plotting
        y: feature to use for y-axis when plotting
    Returns:
        plt: scatter plot
    """
    red_green = sns.diverging_palette(133, 10, l=60, n=2)
    ax = sns.scatterplot(x=x, y=y, data=df, hue=df['diabetes'], palette=red_green)
    return ax


def main():
    """The main function uses argparser to get arguments. Reads csv from file,
    takes in features to plot and plot them in a scatterplot
    with arguments and print out the result (color code formatted text).
    Arguments are described in help.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('csvfile', type=str,
                           help='csv file containing data')
    argparser.add_argument('printTestData', type=bool,
                           help='Prints out tests')
    args = argparser.parse_args()

    df = readDataFromCSV(args.csvfile)
    variables = list(df.keys())[:-1]  # Won't allow glucose as argument
    print("Variables:", end=' ')
    print(*variables, sep=", ")
    x = input("Use variable for x: ")
    y = input("Use variable for y: ")
    if x and y in variables:
        scatterplot(df, x, y)
        plt.show()
    else:
        raise ValueError("Variables input does not exist, plotting aborted")


if __name__ == '__main__':
    main()
