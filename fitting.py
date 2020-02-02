#!/usr/bin/env python3
import argparse
import pandas as pd
from data import data, readDataFromCSV
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from typing import Type


def fit(training_data: pd.DataFrame, *features: str, classifier: str = "knn",
        **kwargs) -> Type[ClassifierMixin]:
    """Function that takes inn training dataframe, features,
    classifier type and classifier arguments. The function fit
    the chosen classifier and returns the fitted classifier.

    Args:
        training_data: dataframe training
        *features: features used for training
        classifier: a string defining which classifier to use.
        **kwargs: a dictionary based on optional arguments, which can be
            used for arguments for the classifiers.

    Returns:
        clf: a fitted classifier
    """

    # Organizing data based on chosen features
    t_data = training_data[[*features]] # unpack tuple
    t_target = training_data["diabetes"].replace(["neg", "pos"], [0, 1])

    # Select classifier type
    if (classifier == "KNN"):
        clf = KNeighborsClassifier(n_neighbors=2)
    elif (classifier == "Logistic Regression"):
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=kwargs["max_iter"])
    elif (classifier == "Linar SVC"):
        clf = LinearSVC(loss="hinge", C=1.0, max_iter=kwargs["max_iter"])
    elif (classifier == "SVC"):
        clf = SVC(gamma='scale', max_iter=kwargs["max_iter"])
    else:
        clf = KNeighborsClassifier(n_neighbors=2)

    return clf.fit(t_data, t_target)


def main():
    """The main function uses argparser to get arguments. Reads csv from file,
    takes in features to plot and prints the predicted accuracy.
    Arguments are described in help.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('csvfile', type=str,
                           help='csv file containing data')
    argparser.add_argument('--features', '-f', nargs='+',
                           help='Features in dataset', required=True)
    argparser.add_argument('--classifier', '-c', type=str,
                           help='classifier type: KNN, Logistic Regression, '
                                'Linar SVC, or SVC. If not given, all are tested.')
    args = argparser.parse_args()

    df = readDataFromCSV(args.csvfile)
    training_data, validation_data = data(df)

    if args.classifier:
        args.classifier = [args.classifier]
    else:
        args.classifier = ["KNN", "Logistic Regression", "Linar SVC", "SVC"]

    # Print accuracy results
    print("\nAccuracy using features: ", end='')
    print(*args.features, sep=', ')

    for classifier in args.classifier:
        clf = fit(training_data, *args.features, classifier=classifier, max_iter=5000)
        prediction = clf.predict(validation_data[[*args.features]])
        accuracy = accuracy_score(validation_data['diabetes'], prediction, normalize=True)
        print('{:<21s}{:>.2f}%'.format(classifier + ':', accuracy*100))



if __name__ == '__main__':
    main()
