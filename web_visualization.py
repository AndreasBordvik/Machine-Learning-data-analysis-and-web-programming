#!/usr/bin/env python3
import os
from flask import Flask
from flask import render_template
from flask import request
from fitting import fit
from sklearn import metrics
from visualize import visualize_clf
from data import *
import time

app = Flask(__name__)


def fitPredPlot(t: pd.DataFrame, v: pd.DataFrame, feature1: str, feature2: str,
                include_error: bool = False, cf="knn"):
    """Overall function that takes inn training dataframe, validation
    dataframe. Fit's the classifier, does predictions based on
    validation data, and plots the true values features,
    classifier type and classifier arguments. The function fit
    the chosen classifier and returns the fitted classifier.

    Args:
        t: dataframe training
        v: dataframe validation
        feature1: feature used for x
        feature2: feature used for y
        include_error: flag used for visualize prediction errors in plot
        cf: The classifier to use for prediction and fitting.

    Returns:
        plt (matplotLib plot): The prediction plot
        acc (integer): Accuracy score for the prediction
    """
    # Organizing data based on choosen features
    v_data = v[[feature1, feature2]]
    v_target = v["diabetes"].replace(["neg", "pos"], [0, 1])

    scatterplot(v, feature1, feature2)

    # Training and predicting
    clf = fit(t, feature1, feature2, classifier=cf, max_iter=5000)
    pred_target = clf.predict(v_data)

    # plot
    plt = visualize_clf(feature1, feature2, v_data,
                        v_target, pred_target, include_error, clf)
    acc = metrics.accuracy_score(v_target, pred_target)
    print(f"Accuracy score for {cf}:{acc}")
    return plt, acc


@app.route("/")
def start():
    """Start function for the initial web page
    Returns:
        - render_template() (function): A function that renders the
            web1.html page, load the start image (url), displays a list
            of possible classfiers, and features
    """
    clfs = ['KNN', 'Logistic Regression', 'Linar SVC', 'SVC']
    featureTypes = ['pregnant', 'glucose', 'pressure', 'triceps',
                    'insulin', 'mass', 'pedigree', 'age', 'None']

    return render_template('web1.html',
                           name='start',
                           url='\static\diabetes\start.png',
                           classifiers=clfs,
                           featureTypes=featureTypes)


@app.route('/plot', methods=['POST'])
def plot():
    """Start function that is initiated from the plot button, and
        the function responds to POST request. The function takes inn
        x-feature,y-feature,errorplotting,and classifier type, from the
        user and does prediction, plots and displays the plot as an image
        on the web page, together with som user frendly information.
    Returns:
        - render_template() (function): A function that renders the
            web1.html page, load the start image (url), displays a list
            of possible classfiers, and features, displays the prediction
            accuracy, and the used classifier.
    """

    assert request.method == 'POST'  # Checks that the code is in POST request
    saveFile = f"{os.getcwd()}/static/diabetes/diabetes"
    tid = time.strftime("%Y-%m-%d_%H_%M_%S")
    saveFile += f"{tid}.png"

    clfs = ['KNN', 'Logistic Regression', 'Linar SVC', 'SVC']
    featureTypes = ['pregnant', 'glucose', 'pressure', 'triceps',
                    'insulin', 'mass', 'pedigree', 'age', 'None']

    # Data
    df = readDataFromCSV("diabetes.csv")
    training, validation = data(df)

    # Acces the form data:
    features = set(request.form.getlist("feat"))
    clf = request.form.get("classifiers")
    plot_error = request.form.get("plot_error")

    if ('None' in features):
        features.remove('None')
    featuresChoosen = list(features)
    if (len(features) == 2):
        newFile = f'\static\diabetes\diabetes{tid}.png'
        newName = f'diabetes{tid}'
    else:
        newName = 'start'
        newFile = '\static\diabetes\start.png'

    if (len(features) >= 2):
        x1 = features.pop()
        x2 = features.pop()
    else:
        x1 = features.pop()
        x2 = "mass"  # Dummy feature

    plot, accuracy = fitPredPlot(training, validation, x1, x2, plot_error, clf)
    plot.savefig(saveFile)
    return render_template('web1.html',
                           name=newName,
                           url=newFile,
                           z=accuracy,
                           classifiers=clfs,
                           clf=clf,
                           featureTypes=featureTypes,
                           features=featuresChoosen)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
