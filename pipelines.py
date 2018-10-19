#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : pipelines.py
    # Creation Date : Mit 22 Aug 2018 18:29:57 CEST
    # Last Modified : Don 23 Aug 2018 15:37:29 CEST
    # Description :
"""
#==============================================================================

import numpy as np
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


class multipleOneHot(BaseEstimator, TransformerMixin):
    """ Basically a wrapper of sklearn.OneHotEncoder for multiple features.
    """

    def __init__(self, features, overwrite):
        self.features = features
        self.overwrite = overwrite

    def fit(self, X):
        self.encDict = {}
        for feat in self.features:
            lab_enc = LabelEncoder()
            x = lab_enc.fit_transform(X[feat])
            one_enc = OneHotEncoder(sparse=False)
            one_enc.fit(x.reshape(-1,1))
            self.encDict[feat] = [lab_enc, one_enc]

        return self

    def transform(self, X):
        for feat in self.features:
            lab_enc = self.encDict[feat][0]
            one_enc = self.encDict[feat][1]
            transformed = lab_enc.transform(X[feat])
            transformed = one_enc.transform(transformed.reshape(-1,1))
            for i, categorical in enumerate(lab_enc.classes_):
                X[str(feat)+"_"+str(categorical)] = transformed[:, i]
            if self.overwrite:
                X.drop(feat, inplace=True, axis=1)

        return X


class multipleStandardScalar(BaseEstimator, TransformerMixin):
    """ Basically a wrapper of sklearn.StandardScaler for multiple features.
    """

    def __init__(self, features, overwrite=False):
        self.features = features
        self.overwrite = overwrite

    def fit(self, X):
        self.scalerDict = {}
        for feat in self.features:
            enc = StandardScaler()
            enc.fit(X[feat].values.reshape(-1,1))
            self.scalerDict[feat] = enc

        return self

    def transform(self, X):
        for feat in self.features:
            if self.overwrite:
                X[feat] = self.scalerDict[feat].transform(X[feat].values.reshape(-1,1))
            else:
                X["standardized_"+str(feat)] = X[feat].values.copy()
                X["standardized_"+str(feat)] = self.scalerDict[feat].transform(
                        X[feat].values.reshape(-1,1)
                        )

        return X


class Binarizer(BaseEstimator, TransformerMixin):
    """ Performs a binary cut on certain discrete or continuous features.

    The input during initialization has to be a dictionary, where the keys indicate
    a column name and the value is a list of 4 elements being (in that order):
        - cut-off
        - operation ("==", ">", ...)
        - Positive name (given if operation is True)
        - Negative name (given if operation is False)

    """

    def __init__(self, binaryDict, overwrite=False, integer_encoding=True):
        self.binaryDict = binaryDict
        self.overwrite = overwrite
        self.encoding = integer_encoding

    def fit(self, X):
        return self

    def transform(self, X):
        for key, value in self.binaryDict.items():
            binaryCol = np.repeat(value[2], X.shape[0])
            if value[1] == "==":
                idx = X[key] == value[0]
            elif value[1] == "!=":
                idx = X[key] != value[0]
            elif value[1] == ">":
                idx = X[key] > value[0]
            elif value[1] == ">=":
                idx = X[key] >= value[0]
            elif value[1] == "<":
                idx = X[key] < value[0]
            elif value[1] == "<=":
                idx = X[key] <= value[0]

            binaryCol[~idx] = value[3]

            if self.encoding:
                self.labeler = LabelEncoder()
                binaryCol = self.labeler.fit_transform(binaryCol)

            if self.overwrite:
                X[key] = binaryCol
            else:
                X["binary_"+str(key)] = X[key].values.copy()
                X["binary_"+str(key)] = binaryCol

        return X


class GroupTransformer(BaseEstimator, TransformerMixin):
    """ Groups together entries in a column with new name.

    Give a dictionary where the key is a column in the data. The value of the
    dictionary is another dictionary where the keys are the new names and the value
    is a list of values which should be replaced.
    Example :   replaceColumns = {
	"education": {"HighEducation": ["Doctorate", "Prof-school", "Master"],
			"Assoc": ["Assoc-acdm", "Assoc-voc"]},
	"marital-status" : {"Absent": ["Married-spouse-absent", "Separated", "Widowed"]}
	}

    Every item in a list gets then substituted by its key in the corresponding column.
    """
    def __init__(self, replaceColumns, overwrite=False):
        self.replacer = {}
        for col, replace in replaceColumns.items():
            replaceDict = {}
            for group, ungrouped in replace.items():
                for singleItem in ungrouped:
                    replaceDict[singleItem] = group
            self.replacer[col] = replaceDict
        self.overwrite = overwrite

    def fit(self, X):
        return self

    def transform(self, X):
        for column, replacement in self.replacer.items():
            if self.overwrite:
                X[column].replace(replacement, inplace=True)
            else:
                X["grouped_"+str(column)] = X[column].values.copy()
                X["grouped_"+str(column)].replace(replacement, inplace=True)

        return(X)


class Logtransform(BaseEstimator, TransformerMixin):
    """ Basic logtransforamtion on certain features
    """

    def __init__(self, features, overwrite=False):
        self.features = features
        self.overwrite = overwrite

    def fit(self, X):
        return self

    def transform(self, X):
        for feat in self.features:
            if self.overwrite:
                X[feat] = np.log(X[feat].values+1)
            else:
                X["log_"+str(feat)] = X[feat].values.copy()
                X["log_"+str(feat)] = np.log(X[feat].values+1)

        return X


class FeatureRemover(BaseEstimator, TransformerMixin):
    """Drops columns from dataframe as a pipeline.
    """

    def __init__(self, features):
        self.features = features

    def fit(self, X):
        return self

    def transform(self, X):
        X.drop(self.features, inplace=True)

        return X
