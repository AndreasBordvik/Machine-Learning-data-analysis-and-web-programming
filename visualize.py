#!/usr/bin/env python3
from typing import Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.base import ClassifierMixin


def _colorList(pred_target: pd.DataFrame, true_target: pd.DataFrame, errorColoring: bool = False):
    """Function that returns a list of strings for color codes
    based on predicted values. The colors are used for
    coloring the plot.

    Args:
        pred_target: dataframe for the predicted response
        true_target: dataframe for the true response
        errorColoring : Applies error colors to the plot
            if True. False only applies green (diabetes negative)
            and red (diabetes positive)

    Returns:
        - colors (list): A list of color strings
        - count (int): The number of colored data points
        - error (int): The number of wrong predictions (error)

    """
    count = 1
    errors = 0
    colors = []
    for pred, true in zip(pred_target, true_target):
        if pred != true:
            if (pred == 0):
                colors.append("b") if errorColoring else colors.append("g")
            else:
                colors.append("m") if errorColoring else colors.append("r")
            errors += 1
        else:
            colors.append("g") if pred == 0 else colors.append("r")
        count += 1
    return colors, count, errors


def visualize_clf(feature1: str, feature2: str, v_data: pd.DataFrame, true_target: pd.DataFrame,
                  pred_target: pd.DataFrame, include_error: bool, clf: Type[ClassifierMixin]):
    """Function that takes inn two features validation data, and classifier. Then
    creates the colors for plotting, and plots the prediction both as scatter plot
    and as pcolormesh plot for coloring of the classification areas.

    Args:
        feature1: features used for x
        feature2: features used for y
        v_data: Dataframe for validation data set
        true_target: Dataframe for the true response
        pred_target: Dataframe for the predicted response
        include_error: Applies error colors to the plot if True
        clf: The classifier
    Returns:
        - pl: The matplotLib plot
    """
    colors, count, errors = _colorList(pred_target, true_target, include_error)

    # Predicting with test data
    x_min, x_max = float(v_data[feature1].min()), float(v_data[feature1].max())
    y_min, y_max = float(v_data[feature2].min()), float(v_data[[feature2]].max())
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    pred_mesh = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    pred_mesh = pred_mesh.reshape(xx.shape)

    fig = plt.figure()
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.pcolormesh(xx, yy, pred_mesh, cmap=cmap_light)

    plt.scatter(v_data[feature1],
                v_data[feature2],
                alpha=0.6,
                c=colors,
                marker='o',
                s=20,
                cmap=cmap_bold)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    neg = mpatches.Patch(color='g', label='Diab. - Neg')
    pos = mpatches.Patch(color='r', label='Diab. - Pos')
    negError = mpatches.Patch(color='b', label='Diab. - Neg(error)')
    posError = mpatches.Patch(color='m', label='Diab - Pos(error)')
    elements = [neg, pos, negError, posError] if include_error else [neg, pos]
    plt.legend(handles=elements)
    plt.axis('tight')
    return fig
