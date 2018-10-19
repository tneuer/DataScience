#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : dataUtils.py
    # Creation Date : Don 23 Aug 2018 16:34:43 CEST
    # Last Modified : Son 26 Aug 2018 15:30:21 CEST
    # Description : Common utilities needed in Data Science
"""
#==============================================================================

import re
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

class FeatureImportance():
    """ Determine feature importance.

    Feature importance assessment is an important task in data science. This class
    provides several methods to determine this importance.
    1) XGB: Boostes decision trees allow for easy in straight forward determination of
        feature importance by counting the number of times the features was used to
        perform a cut.
    2) PCA: Principal Components analysis determines which feature is responsible for the
        most amount of explained variance.
    """

    def __init__(self, method="xgb", parameter_dict=None, *args, **kwargs):
        """
        Parameters
        ----------
        method : str
            Has to be in ["xgb", "PCA"] or an error will be raised. See class documentation
            for more information.
        parameter_dict : dict
            Keyword arguments for chosen method.
        """
        self.available = ["xgb", "PCA"]
        if method not in self.available:
            raise ValueError("'method' has to be one of {}.".format(self.available))

        if parameter_dict is not None and kwargs:
            raise TypeError("Either parameters in one dict or kwargs, not both.")
        elif parameter_dict is None:
            parameter_dict = kwargs

        if method == "xgb":
            model = xgb.XGBClassifier(**parameter_dict)
        if method == "PCA":
            if "n_components" in parameter_dict:
                raise Warning("'n_compnents' should not be handed as parameter during"+
                        "feature importance analysis.")
            model = PCA(**parameter_dict)

        self.method = method
        self.model = model
        self.trained = False


    def fit(self, X, y=None, **kwargs):
        if self.method == "xgb":
            try: #Check if one-Hot encoded, not valid for xgb
                y = np.array(y)
                y.shape[1]
                y = np.argmax(y, axis=1)
            except IndexError:
                pass
            self.model.fit(X, y, **kwargs)
            pred = self.model.predict(X)
            self.accuracy = np.mean(np.equal(pred, y))
        elif self.method == "PCA":
            self.model.fit(X, **kwargs)

        if isinstance(X, pd.DataFrame): # feature names given as columns?
            self.featureNames = X.columns.values
            self.namesGiven = True
        else:
            self.featureNames = np.arange(X.shape[1])
            self.namesGiven = False
        self.n_instances = X.shape[0]
        self.trained = True


    def assess_importance(self, plot=True, n_best=20):
        """
        Parameters
        ----------
        plot : bool
            If True, a figure object is returned, which can be plotted or saved.
        n_best : int
            Only used if plot==True, number of features shown in the importance plot.
        """
        fitType = "'fit(X, y, **kwargs)'" if self.method=="xgb" else "'fit(X, **kwargs)'"
        assert self.trained, "Model not yet trained. First call {} method".format(fitType)

        if self.method == "xgb":
            importance = self.xgb_importance(plot=plot, n_best=n_best)
        elif self.method == "PCA":
            importance = self.pca_importance(plot=plot, n_best=n_best)

        if plot:
            plt.plot()
            return importance[0], importance[1]
        else:
            return importance[0], importance[1]


    def xgb_importance(self, plot=True, n_best=20):
        """ Supervised feature importance assessment for a specific goal.

        Uses the XGBoost algorithm to determine the best feature for a classification/
        regression task at hand.
        """
        fImportance = self.model._Booster.get_fscore()
        fNames = []
        values = []
        for key, value in fImportance.items():
            fNames.append(key)
            values.append(value)

        sortInd = np.argsort(values)[::-1]
        values = np.array(values)[sortInd]
        fNames = np.array(fNames)[sortInd]

        if self.namesGiven:
            importance = np.array([(f, v) for f,v in zip(fNames, values)])
        else:
            importance = np.array([(int(f[1:]), v) for f,v in zip(fNames, values)])

        if plot:
            values = values[:n_best]
            fNames = fNames[:n_best]

            fig = plt.figure()
            bars = plt.bar(np.arange(len(values)), values)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                width = bar.get_x() + bar.get_width()/2
                plt.text(width, height, values[i], ha="center", va="bottom")
            plt.xticks(np.arange(len(values)), fNames, rotation=70, size=9)
            plt.ylabel("F-score")
            plt.title("#Instances: {}, Accuracy: {}".format(self.n_instances,
                                                            self.accuracy))

            return importance, fig
        else:
            return importance


    def pca_importance(self, plot=True, n_best=20):
        """ Unsupervised feature importance assessment technique.

        Uses the scikit-learn Principal components analysis technique in order to find
        the feature explaining the largest amount of variance.
        Basically sorted by variance.
        """
        lamda = self.model.explained_variance_ratio_
        csum = np.cumsum(lamda)

        fig1 = plt.figure()
        plt.plot(csum, marker="o", label="Explained Variance")
        plt.xlabel("#Features"); plt.ylabel("Explained variance")
        plt.ylim([0, 1.1]); plt.yticks(np.arange(0,1,0.1))
        plt.grid()

        fractions = [0.99, 0.95, 0.90]
        ds = [np.argmax(csum >= f) for f in fractions]
        titlestring = "Explained variance: #features\n|   "
        for f, d in zip(fractions, ds):
            titlestring += "{}: - {}   |    ".format(f, d)
            plt.axvline(x=d, c="red", linewidth=2, zorder=0, linestyle="--")
            plt.axhline(y=f, c="red", linewidth=2, zorder=0, linestyle="--")
        plt.title(titlestring)

        importance = self.model.components_
        importance = [imp**2 * l for l, imp in zip(lamda, importance)]
        importance = np.sum(importance, axis=0)

        sortInd = np.argsort(importance)[::-1]
        values = np.round(np.array(importance)[sortInd], 3)
        fNames = self.featureNames[sortInd]

        importance = np.array([(f, v) for f,v in zip(fNames, values)])


        if plot:
            values = values[:n_best]
            fNames = fNames[:n_best]

            fig2 = plt.figure()
            bars = plt.bar(np.arange(len(values)), values)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                width = bar.get_x() + bar.get_width()/2
                plt.text(width, height, values[i], ha="center", va="bottom")
            plt.xticks(np.arange(len(values)), fNames, rotation=70, size=9)
            plt.ylabel("Score")
            plt.title("#Instances: {}".format(self.n_instances))
            return importance, (fig1, fig2)
        else:
            return importance


if __name__ == "__main__":
    train_examples = 1000
    print("Load training data...")
    train_x = pd.read_csv("./Processed_train.csv")
    train_x = train_x.iloc[:train_examples, :]

    with open("./Train_Label.pickle", "rb") as f:
        train_y = pickle.load(f)
        train_y = train_y[:train_examples, :]

    features = train_x.columns.values
    used_features = features[:]
    train_x = train_x[used_features]
    # train_x = train_x[used_features].values

    # parameters = {"max_depth":5, "n_estimator":10, "silent":True}
    # parameters = {"n_components":43}
    imp = FeatureImportance(method="xgb")
    imp.fit(train_x, train_y)
    importance, figs = imp.assess_importance(plot=True)

    plt.show()
