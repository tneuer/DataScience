


# OVERVIEW

Following descriptions are provided:
1. [Module: dataUtils.py](#module:-----datautils.py)
2. [Module: ModelAnalysis.py](#module:-----modelanalysis.py)
3. [Module: pipelines.py](#module:-----pipelines.py)


# Module:     dataUtils.py

This module contains:
- [1 Classes](#classes-of-dataUtils.py)
- [0 Functions](#functions-of-dataUtils.py)

| General                      | Value|
| ---------------------------- | ---- |
| Lines of Code (LoC)          | 0231 |
| Lines of documentation (LoD) | 0050 |
| Empty lines (LoN)            | 0042 |
| Number of classes (NoC)      | 0001 |
| Number of functions (NoF)    | 0000 |

## Documentation

|3| """
|4| # Author : Thomas Neuer (tneuer)
|5| # File Name : dataUtils.py
|6| # Creation Date : Don 23 Aug 2018 16:34:43 CEST
|7| # Last Modified : Fre 26 Okt 2018 17:54:47 CEST
|8| # Description : Common utilities needed in Data Science
|9| """



## Imports

Following packages are imported:

| Package                          | Imported as      | Imported objects                    |
| -------------------------------- | ---------------- | ----------------------------------- |
| re                               | -                | -                                   |
| pickle                           | -                | -                                   |
| numpy                            | np               | -                                   |
| pandas                           | pd               | -                                   |
| xgboost                          | xgb              | -                                   |
| matplotlib                       | plt              | -                                   |
| sklearn                          | -                | PCA                                 |


# Classes of dataUtils.py
This module contains following classes:
- [FeatureImportance                        (187, 38)](#class:-featureimportance)




#### Class: FeatureImportance

Jump to:
- [Methods](#methods-of-featureimportance)
- [Attributes](#attributes-of-featureimportance)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0022 |
| End line (End)               | 0210 |
| Lines of Code (LoC)          | 0187 |
| Lines of documentation (LoD) | 0038 |
| Empty lines (LoN)            | 0033 |
| Number of methods            | 0005 |
| Number of Attributes         | 0008 |
| Number of parents            | 0001 |

##### Documentation

|23| """ Determine feature importance.
|24| 
|25| Feature importance assessment is an important task in data science. This class
|26| provides several methods to determine this importance.
|27| 1) XGB: Boostes decision trees allow for easy in straight forward determination of
|28| feature importance by counting the number of times the features was used to
|29| perform a cut.
|30| 2) PCA: Principal Components analysis determines which feature is responsible for the
|31| most amount of explained variance.
|32| """



##### Inheritance

This class inherits from:



##### Methods of FeatureImportance

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [assess\_importance](#method:-assess\_importance)
4. [xgb\_importance](#method:-xgb\_importance)
5. [pca\_importance](#method:-pca\_importance)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0034 |
| End line (End)               | 0066 |
| Lines of Code (LoC)          | 0031 |
| Lines of documentation (LoD) | 0009 |
| Empty lines (LoN)            | 0005 |

##### Documentation

|35| """
|36| Parameters
|37| ----------
|38| method : str
|39| Has to be in ["xgb", "PCA"] or an error will be raised. See class documentation
|40| for more information.
|41| parameter_dict : dict
|42| Keyword arguments for chosen method.
|43| """


##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| method                         | "xgb"                          |
| parameter_dict                 | None                           |
| *args                          |                                |
| **kwargs                       |                                |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0066 |
| End line (End)               | 0090 |
| Lines of Code (LoC)          | 0023 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0003 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |
| y                              | None                           |
| **kwargs                       |                                |

##### Returns

#### Method: assess\_importance

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0090 |
| End line (End)               | 0114 |
| Lines of Code (LoC)          | 0023 |
| Lines of documentation (LoD) | 0008 |
| Empty lines (LoN)            | 0004 |

##### Documentation

|91| """
|92| Parameters
|93| ----------
|94| plot : bool
|95| If True, a figure object is returned, which can be plotted or saved.
|96| n_best : int
|97| Only used if plot==True, number of features shown in the importance plot.
|98| """


##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| plot                           | True                           |
| n_best                         | 20                             |

##### Returns

- Return 1:  importance[0]  &  importance[1]
- Return 2:  importance[0]  &  importance[1]
#### Method: xgb\_importance

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0114 |
| End line (End)               | 0156 |
| Lines of Code (LoC)          | 0041 |
| Lines of documentation (LoD) | 0005 |
| Empty lines (LoN)            | 0008 |

##### Documentation

|115| """ Supervised feature importance assessment for a specific goal.
|116| 
|117| Uses the XGBoost algorithm to determine the best feature for a classification/
|118| regression task at hand.
|119| """


##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| plot                           | True                           |
| n_best                         | 20                             |

##### Returns

- Return 1:  importance  &  fig
- Return 2:  importance
#### Method: pca\_importance

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0156 |
| End line (End)               | 0210 |
| Lines of Code (LoC)          | 0053 |
| Lines of documentation (LoD) | 0006 |
| Empty lines (LoN)            | 0011 |

##### Documentation

|157| """ Unsupervised feature importance assessment technique.
|158| 
|159| Uses the scikit-learn Principal components analysis technique in order to find
|160| the feature explaining the largest amount of variance.
|161| Basically sorted by variance.
|162| """


##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| plot                           | True                           |
| n_best                         | 20                             |

##### Returns

- Return 1:  importance  &  (fig1  &  fig2)
- Return 2:  importance


##### Attributes of FeatureImportance

A list of the used attributes:

0 trained; 1 method; 2 available; 3 namesGiven; 4 n_instances; 5 model; 
6 accuracy; 7 featureNames; 






# Functions of dataUtils.py
This module contains following functions:







# Module:     ModelAnalysis.py

This module contains:
- [0 Classes](#classes-of-ModelAnalysis.py)
- [2 Functions](#functions-of-ModelAnalysis.py)

| General                      | Value|
| ---------------------------- | ---- |
| Lines of Code (LoC)          | 0178 |
| Lines of documentation (LoD) | 0047 |
| Empty lines (LoN)            | 0031 |
| Number of classes (NoC)      | 0000 |
| Number of functions (NoF)    | 0002 |

## Documentation

|3| """
|4| # Author : Thomas Neuer (tneuer)
|5| # File Name : ModelAnalysis.py
|6| # Creation Date : Fre 26 Okt 2018 17:59:05 CEST
|7| # Last Modified : Sam 27 Okt 2018 22:15:47 CEST
|8| # Description : Some utilities to help analyze a given model.
|9| """



## Imports

Following packages are imported:

| Package                          | Imported as      | Imported objects                    |
| -------------------------------- | ---------------- | ----------------------------------- |
| pickle                           | -                | -                                   |
| numpy                            | np               | -                                   |
| pandas                           | pd               | -                                   |
| keras                            | -                | load_model                          |
| matplotlib                       | plt              | -                                   |


# Classes of ModelAnalysis.py
This module contains following classes:






# Functions of ModelAnalysis.py
This module contains following functions:
- [plot\_confidence\_in\_true\_label        (66, 19)](#function:-plot\_confidence\_in\_true\_label)
- [plot\_confidence\_in\_predicted\_label   (66, 19)](#function:-plot\_confidence\_in\_predicted\_label)




#### Function: plot\_confidence\_in\_true\_label

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0021 |
| End line (End)               | 0088 |
| Lines of Code (LoC)          | 0066 |
| Lines of documentation (LoD) | 0019 |
| Empty lines (LoN)            | 0009 |

##### Documentation

|22| """ Plot the confidence of the input separated by true class label.
|23| 
|24| If predictions are given, the other three keyword arguments must be given in order
|25| to be able to calculate these predictions. If predictions is not None, the other three
|26| inputs are ignored.
|27| 
|28| Arguments
|29| ---------
|30| predProba : np.ndarray [None]
|31| Array of dimension (nr_examples, nr_classes) where each entry is the confidence
|32| in the different classes.
|33| data : pd.DataFrame or np.ndarray [None]
|34| Contains data in shape (nr_examples, nr_features). Ignored if predictions are given.
|35| labels : pd.Series, np.ndarray or list of integers
|36| Contains the labels in shape (nr_examples, ) where each class is represented by an index,
|37| e.g.: [0,1,0,0,3,4,1,...] for 4 (or more) classes
|38| model : model [None]
|39| Trained model, which has a method called predcit_proba to predict the class confidence.
|40| """


##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| labels                         |                                |
| predProba                      | None                           |
| data                           | None                           |
| model                          | None                           |

##### Returns

- Return 1:  fig



#### Function: plot\_confidence\_in\_predicted\_label

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0088 |
| End line (End)               | 0155 |
| Lines of Code (LoC)          | 0066 |
| Lines of documentation (LoD) | 0019 |
| Empty lines (LoN)            | 0009 |

##### Documentation

|89| """ Plot the confidence of the input separated by predicted class label.
|90| 
|91| If predictions are given, the other three keyword arguments must be given in order
|92| to be able to calculate these predictions. If predictions is not None, the other three
|93| inputs are ignored.
|94| 
|95| Arguments
|96| ---------
|97| predProba : np.ndarray [None]
|98| Array of dimension (nr_examples, nr_classes) where each entry is the confidence
|99| in the different classes.
|100| data : pd.DataFrame or np.ndarray [None]
|101| Contains data in shape (nr_examples, nr_features). Ignored if predictions are given.
|102| labels : pd.Series, np.ndarray or list of integers
|103| Contains the labels in shape (nr_examples, ) where each class is represented by an index,
|104| e.g.: [0,1,0,0,3,4,1,...] for 4 (or more) classes
|105| model : model [None]
|106| Trained model, which has a method called predcit_proba to predict the class confidence.
|107| """


##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| labels                         |                                |
| predProba                      | None                           |
| data                           | None                           |
| model                          | None                           |

##### Returns

- Return 1:  fig






# Module:     pipelines.py

This module contains:
- [7 Classes](#classes-of-pipelines.py)
- [0 Functions](#functions-of-pipelines.py)

| General                      | Value|
| ---------------------------- | ---- |
| Lines of Code (LoC)          | 0300 |
| Lines of documentation (LoD) | 0061 |
| Empty lines (LoN)            | 0060 |
| Number of classes (NoC)      | 0007 |
| Number of functions (NoF)    | 0000 |

## Documentation

|3| """
|4| # Author : Thomas Neuer (tneuer)
|5| # File Name : pipelines.py
|6| # Creation Date : Mit 22 Aug 2018 18:29:57 CEST
|7| # Last Modified : Son 28 Okt 2018 10:41:19 CET
|8| # Description :
|9| """



## Imports

Following packages are imported:

| Package                          | Imported as      | Imported objects                    |
| -------------------------------- | ---------------- | ----------------------------------- |
| pickle                           | -                | -                                   |
| numpy                            | np               | -                                   |
| pandas                           | pd               | -                                   |
| collections                      | -                | Counter                             |
| sklearn                          | -                | Pipeline, FeatureUnion              |
| sklearn                          | -                | BaseEstimator, TransformerMixin     |
| sklearn                          | -                | OneHotEncoder, StandardScaler, LabelEncoder |


# Classes of pipelines.py
This module contains following classes:
- [multipleOneHot                           (32, 2)](#class:-multipleonehot)
- [multipleStandardScalar                   (29, 2)](#class:-multiplestandardscalar)
- [Binarizer                                (50, 10)](#class:-binarizer)
- [GroupTransformer                         (37, 13)](#class:-grouptransformer)
- [Logtransform                             (21, 2)](#class:-logtransform)
- [RareConstructor                          (78, 21)](#class:-rareconstructor)
- [FeatureRemover                           (14, 2)](#class:-featureremover)




#### Class: multipleOneHot

Jump to:
- [Methods](#methods-of-multipleonehot)
- [Attributes](#attributes-of-multipleonehot)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0023 |
| End line (End)               | 0056 |
| Lines of Code (LoC)          | 0032 |
| Lines of documentation (LoD) | 0002 |
| Empty lines (LoN)            | 0007 |
| Number of methods            | 0003 |
| Number of Attributes         | 0003 |
| Number of parents            | 0002 |

##### Documentation

|24| """ Basically a wrapper of sklearn.OneHotEncoder for multiple features.
|25| """



##### Inheritance

This class inherits from:
- BaseEstimator
-  TransformerMixin



##### Methods of multipleOneHot

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [transform](#method:-transform)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0027 |
| End line (End)               | 0031 |
| Lines of Code (LoC)          | 0003 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| features                       |                                |
| overwrite                      |                                |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0031 |
| End line (End)               | 0042 |
| Lines of Code (LoC)          | 0010 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0002 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  self
#### Method: transform

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0042 |
| End line (End)               | 0056 |
| Lines of Code (LoC)          | 0013 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0003 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  X


##### Attributes of multipleOneHot

A list of the used attributes:

0 encDict; 1 overwrite; 2 features; 




#### Class: multipleStandardScalar

Jump to:
- [Methods](#methods-of-multiplestandardscalar)
- [Attributes](#attributes-of-multiplestandardscalar)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0056 |
| End line (End)               | 0086 |
| Lines of Code (LoC)          | 0029 |
| Lines of documentation (LoD) | 0002 |
| Empty lines (LoN)            | 0007 |
| Number of methods            | 0003 |
| Number of Attributes         | 0003 |
| Number of parents            | 0002 |

##### Documentation

|57| """ Basically a wrapper of sklearn.StandardScaler for multiple features.
|58| """



##### Inheritance

This class inherits from:
- BaseEstimator
-  TransformerMixin



##### Methods of multipleStandardScalar

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [transform](#method:-transform)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0060 |
| End line (End)               | 0064 |
| Lines of Code (LoC)          | 0003 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| features                       |                                |
| overwrite                      | False                          |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0064 |
| End line (End)               | 0073 |
| Lines of Code (LoC)          | 0008 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0002 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  self
#### Method: transform

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0073 |
| End line (End)               | 0086 |
| Lines of Code (LoC)          | 0012 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0003 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  X


##### Attributes of multipleStandardScalar

A list of the used attributes:

0 scalerDict; 1 overwrite; 2 features; 




#### Class: Binarizer

Jump to:
- [Methods](#methods-of-binarizer)
- [Attributes](#attributes-of-binarizer)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0086 |
| End line (End)               | 0137 |
| Lines of Code (LoC)          | 0050 |
| Lines of documentation (LoD) | 0010 |
| Empty lines (LoN)            | 0011 |
| Number of methods            | 0003 |
| Number of Attributes         | 0004 |
| Number of parents            | 0002 |

##### Documentation

|87| """ Performs a binary cut on certain discrete or continuous features.
|88| 
|89| The input during initialization has to be a dictionary, where the keys indicate
|90| a column name and the value is a list of 4 elements being (in that order):
|91| - cut-off
|92| - operation ("==", ">", ...)
|93| - Positive name (given if operation is True)
|94| - Negative name (given if operation is False)
|95| 
|96| """



##### Inheritance

This class inherits from:
- BaseEstimator
-  TransformerMixin



##### Methods of Binarizer

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [transform](#method:-transform)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0098 |
| End line (End)               | 0103 |
| Lines of Code (LoC)          | 0004 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| binaryDict                     |                                |
| overwrite                      | False                          |
| integer_encoding               | True                           |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0103 |
| End line (End)               | 0106 |
| Lines of Code (LoC)          | 0002 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  self
#### Method: transform

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0106 |
| End line (End)               | 0137 |
| Lines of Code (LoC)          | 0030 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0006 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  X


##### Attributes of Binarizer

A list of the used attributes:

0 binaryDict; 1 encoding; 2 overwrite; 3 labeler; 




#### Class: GroupTransformer

Jump to:
- [Methods](#methods-of-grouptransformer)
- [Attributes](#attributes-of-grouptransformer)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0137 |
| End line (End)               | 0175 |
| Lines of Code (LoC)          | 0037 |
| Lines of documentation (LoD) | 0013 |
| Empty lines (LoN)            | 0007 |
| Number of methods            | 0003 |
| Number of Attributes         | 0002 |
| Number of parents            | 0002 |

##### Documentation

|138| """ Groups together entries in a column with new name.
|139| 
|140| Give a dictionary where the key is a column in the data. The value of the
|141| dictionary is another dictionary where the keys are the new names and the value
|142| is a list of values which should be replaced.
|143| Example :   replaceColumns = {
|144| 	"education": {"HighEducation": ["Doctorate", "Prof-school", "Master"],
|145| 			"Assoc": ["Assoc-acdm", "Assoc-voc"]},
|146| 	"marital-status" : {"Absent": ["Married-spouse-absent", "Separated", "Widowed"]}
|147| 	}
|148| 
|149| Every item in a list gets then substituted by its key in the corresponding column.
|150| """



##### Inheritance

This class inherits from:
- BaseEstimator
-  TransformerMixin



##### Methods of GroupTransformer

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [transform](#method:-transform)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0151 |
| End line (End)               | 0161 |
| Lines of Code (LoC)          | 0009 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| replaceColumns                 |                                |
| overwrite                      | False                          |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0161 |
| End line (End)               | 0164 |
| Lines of Code (LoC)          | 0002 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  self
#### Method: transform

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0164 |
| End line (End)               | 0175 |
| Lines of Code (LoC)          | 0010 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0003 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns



##### Attributes of GroupTransformer

A list of the used attributes:

0 replacer; 1 overwrite; 




#### Class: Logtransform

Jump to:
- [Methods](#methods-of-logtransform)
- [Attributes](#attributes-of-logtransform)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0175 |
| End line (End)               | 0197 |
| Lines of Code (LoC)          | 0021 |
| Lines of documentation (LoD) | 0002 |
| Empty lines (LoN)            | 0006 |
| Number of methods            | 0003 |
| Number of Attributes         | 0002 |
| Number of parents            | 0002 |

##### Documentation

|176| """ Basic logtransforamtion on certain features
|177| """



##### Inheritance

This class inherits from:
- BaseEstimator
-  TransformerMixin



##### Methods of Logtransform

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [transform](#method:-transform)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0179 |
| End line (End)               | 0183 |
| Lines of Code (LoC)          | 0003 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| features                       |                                |
| overwrite                      | False                          |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0183 |
| End line (End)               | 0186 |
| Lines of Code (LoC)          | 0002 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  self
#### Method: transform

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0186 |
| End line (End)               | 0197 |
| Lines of Code (LoC)          | 0010 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0003 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  X


##### Attributes of Logtransform

A list of the used attributes:

0 overwrite; 1 features; 




#### Class: RareConstructor

Jump to:
- [Methods](#methods-of-rareconstructor)
- [Attributes](#attributes-of-rareconstructor)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0197 |
| End line (End)               | 0276 |
| Lines of Code (LoC)          | 0078 |
| Lines of documentation (LoD) | 0021 |
| Empty lines (LoN)            | 0009 |
| Number of methods            | 0003 |
| Number of Attributes         | 0004 |
| Number of parents            | 0002 |

##### Documentation

|198| """ Groups categories in a column to a combined rare class.
|199| 
|200| Some categircal features might have to many categories which are either
|201| not important or too under represented to help. This pipelines helps to
|202| remove those categories by combining them into a "rare" class.
|203| """



##### Inheritance

This class inherits from:
- BaseEstimator
-  TransformerMixin



##### Methods of RareConstructor

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [transform](#method:-transform)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0205 |
| End line (End)               | 0240 |
| Lines of Code (LoC)          | 0034 |
| Lines of documentation (LoD) | 0015 |
| Empty lines (LoN)            | 0001 |

##### Documentation

|206| """
|207| Arguments
|208| ---------
|209| features : List or string
|210| Indicates which columns should be cut
|211| cuts : List or None [None]
|212| All categories with less counts than indicated by cuts are combined to the rare class.
|213| If None for a feature, the counts are printed per category to the terminal and
|214| the user can decide on a useful cut.
|215| overwrite : bool [False]
|216| Indicates wether the column in question should be replaced or a new column
|217| is added.
|218| new_cat : -
|219| Value which gets substituted for categories below the threshold
|220| """


##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| features                       |                                |
| cuts                           | None                           |
| new_cats                       | "rare"                         |
| overwrite                      | False                          |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0240 |
| End line (End)               | 0243 |
| Lines of Code (LoC)          | 0002 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  self
#### Method: transform

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0243 |
| End line (End)               | 0276 |
| Lines of Code (LoC)          | 0032 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0005 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  X


##### Attributes of RareConstructor

A list of the used attributes:

0 new_cats; 1 overwrite; 2 cuts; 3 features; 




#### Class: FeatureRemover

Jump to:
- [Methods](#methods-of-featureremover)
- [Attributes](#attributes-of-featureremover)

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0276 |
| End line (End)               | 0291 |
| Lines of Code (LoC)          | 0014 |
| Lines of documentation (LoD) | 0002 |
| Empty lines (LoN)            | 0005 |
| Number of methods            | 0003 |
| Number of Attributes         | 0001 |
| Number of parents            | 0002 |

##### Documentation

|277| """Drops columns from dataframe as a pipeline.
|278| """



##### Inheritance

This class inherits from:
- BaseEstimator
-  TransformerMixin



##### Methods of FeatureRemover

This class contains following methods:

1. [\_\_init\_\_](#method:-\_\_init\_\_)
2. [fit](#method:-fit)
3. [transform](#method:-transform)
#### Method: \_\_init\_\_

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0280 |
| End line (End)               | 0283 |
| Lines of Code (LoC)          | 0002 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| features                       |                                |

##### Returns

#### Method: fit

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0283 |
| End line (End)               | 0286 |
| Lines of Code (LoC)          | 0002 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0001 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  self
#### Method: transform

| General                      | Value|
| ---------------------------- | ---- |
| Start line (Start)           | 0286 |
| End line (End)               | 0291 |
| Lines of Code (LoC)          | 0004 |
| Lines of documentation (LoD) | 0000 |
| Empty lines (LoN)            | 0002 |

##### Documentation

_No documentation available_

##### Arguments

| Arguments                      | Default                        |
| -------------------------------| ------------------------------ |
| self                           |                                |
| X                              |                                |

##### Returns

- Return 1:  X


##### Attributes of FeatureRemover

A list of the used attributes:

0 features; 






# Functions of pipelines.py
This module contains following functions:







