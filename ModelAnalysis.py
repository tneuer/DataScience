#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : ModelAnalysis.py
    # Creation Date : Fre 26 Okt 2018 17:59:05 CEST
    # Last Modified : Sam 27 Okt 2018 22:15:47 CEST
    # Description : Some utilities to help analyze a given model.
"""
#==============================================================================

import pickle

import numpy as np
import pandas as pd

from keras.models import load_model

import matplotlib.pyplot as plt

def plot_confidence_in_true_label(labels, predProba=None, data=None, model=None):
    """ Plot the confidence of the input separated by true class label.

    If predictions are given, the other three keyword arguments must be given in order
    to be able to calculate these predictions. If predictions is not None, the other three
    inputs are ignored.

    Arguments
    ---------
    predProba : np.ndarray [None]
        Array of dimension (nr_examples, nr_classes) where each entry is the confidence
        in the different classes.
    data : pd.DataFrame or np.ndarray [None]
        Contains data in shape (nr_examples, nr_features). Ignored if predictions are given.
    labels : pd.Series, np.ndarray or list of integers
        Contains the labels in shape (nr_examples, ) where each class is represented by an index,
        e.g.: [0,1,0,0,3,4,1,...] for 4 (or more) classes
    model : model [None]
        Trained model, which has a method called predcit_proba to predict the class confidence.
    """
    if predProba is None:
        if model is None or data is None:
            raise ValueError("If predictions is None, a model and the data has to be provided.")
        else:
            if isinstance(data, pd.DataFrame):
                data = data.values
            predProba = model.predict_proba(data)

    nr_classes = len(set(labels))
    if nr_classes > 3:
        raise ValueError("Not implemented for more than 3 classes due to missing colors.")

    pred = np.argmax(predProba, axis=1)
    X = []
    label = []
    color = []
    recall = []
    precision = []

    for i in np.unique(labels):
        i = int(i)
        conf_true_label_i = predProba[labels==i, i]
        true_i = conf_true_label_i[(pred==i)[labels==i]]
        false_i = conf_true_label_i[(pred!=i)[labels==i]]
        TP = (pred==i)[labels==i]
        recall.append(np.round(sum(TP)/sum(labels==i), 2))
        precision.append(np.round(sum(TP)/sum(labels==i), 2))
        X.append(true_i)
        X.append(false_i)
        label.append("True class {} ({})".format(i, recall[-1]))
        label.append("False class {} ({})".format(i, precision[-1]))
        color.append(["#003668", "#8B0000", "#000000"][i])
        color.append(["#7fa1c1", "#d09999", "#7F7F7F"][i])

    fig = plt.figure(figsize=(20,10))
    plt.hist(X, histtype="barstacked", bins=20, label=label, color=color)
    plt.axvline(x=1/nr_classes, color="k", linewidth=3, linestyle="--", alpha=0.6)
    plt.axvline(x=0.5, color="k", linewidth=3, linestyle="--", alpha=0.6)
    plt.legend(fontsize=20)
    plt.xlabel("P(pred)")
    plt.ylabel("Count")
    plt.title("True label vs. confidence (Recall)\n(True: Recall, False: Precision)",fontsize=20)
    plt.xticks(np.arange(0, 1, 0.1))

    return fig


def plot_confidence_in_predicted_label(labels, predProba=None, data=None, model=None):
    """ Plot the confidence of the input separated by predicted class label.

    If predictions are given, the other three keyword arguments must be given in order
    to be able to calculate these predictions. If predictions is not None, the other three
    inputs are ignored.

    Arguments
    ---------
    predProba : np.ndarray [None]
        Array of dimension (nr_examples, nr_classes) where each entry is the confidence
        in the different classes.
    data : pd.DataFrame or np.ndarray [None]
        Contains data in shape (nr_examples, nr_features). Ignored if predictions are given.
    labels : pd.Series, np.ndarray or list of integers
        Contains the labels in shape (nr_examples, ) where each class is represented by an index,
        e.g.: [0,1,0,0,3,4,1,...] for 4 (or more) classes
    model : model [None]
        Trained model, which has a method called predcit_proba to predict the class confidence.
    """
    if predProba is None:
        if model is None or data is None:
            raise ValueError("If predictions is None, a model and the data has to be provided.")
        else:
            if isinstance(data, pd.DataFrame):
                data = data.values
            predProba = model.predict_proba(data)

    nr_classes = len(set(labels))
    if nr_classes > 3:
        raise ValueError("Not implemented for more than 3 classes due to missing colors.")

    pred = np.argmax(predProba, axis=1)
    X = []
    label = []
    color = []
    recall = []
    precision = []

    for i in np.unique(labels):
        i = int(i)
        conf_pred_label_i = predProba[pred==i, i]
        true_i = conf_pred_label_i[(labels==i)[pred==i]]
        false_i = conf_pred_label_i[(labels!=i)[pred==i]]
        TP = (pred==i)[labels==i]
        recall.append(np.round(sum(TP)/sum(labels==i), 2))
        precision.append(np.round(sum(TP)/sum(labels==i), 2))
        X.append(true_i)
        X.append(false_i)
        label.append("True class {} ({})".format(i, recall[-1]))
        label.append("False class {} ({})".format(i, precision[-1]))
        color.append(["#003668", "#8B0000", "#000000"][i])
        color.append(["#7fa1c1", "#d09999", "#7F7F7F"][i])

    fig = plt.figure(figsize=(20,10))
    plt.hist(X, histtype="barstacked", bins=20, label=label, color=color)
    plt.axvline(x=1/nr_classes, color="k", linewidth=3, linestyle="--", alpha=0.6)
    plt.axvline(x=0.5, color="k", linewidth=3, linestyle="--", alpha=0.6)
    plt.legend(fontsize=20)
    plt.xlabel("P(pred)")
    plt.ylabel("Count")
    plt.title("Predicted label vs confidence (Precision)\n[red PREDICTED increase, blue PREDICTED decrease, black PREDICTED unchanged]\n(True: Recall, False: Precision)", fontsize=20)
    plt.xticks(np.arange(0, 1, 0.1))

    return fig


if __name__ == "__main__":
    model = load_model("../NN/TestNet.h5")
    datafolder = "../0TestData/classification/predictSalary"

    print("Load training data...")
    train_x = pd.read_csv("{}/Processed_train.csv".format(datafolder))
    train_x = train_x.iloc[:, :]

    with open("{}/Train_Label.pickle".format(datafolder), "rb") as f:
        train_y = pickle.load(f)
        train_y = train_y[:, 0]

    features = train_x.columns.values
    used_features = features[:20]

    train_x = train_x[used_features]
    train_x = train_x.values

    pred = model.predict(train_x)

    fig1 = plot_confidence_in_predicted_label(predProba=pred, labels=train_y)
    fig2 = plot_confidence_in_true_label(predProba=pred, labels=train_y)

    plt.show()
