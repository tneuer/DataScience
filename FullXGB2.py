"""
    # Author : Thomas Neuer (tneuer)
    # File Name : DataScience / FullXGB.py
    # Creation Date : 11.11.18 11:27 CET
    # Description : Full XGB analysis developed in EPha.ch. Designed for classification.
"""
# ==============================================================================

import os
import re
import time
import pickle
import shutil
import xgboost
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from collections import defaultdict, Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from ModelAnalysis import plot_confidence_in_predicted_label, plot_confidence_in_true_label

plt.style.use('seaborn-white')


def savefig(list_of_plots, path):
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(path)
    for fig in list_of_plots:
        pdf.savefig(fig)
    pdf.close()


class Node():
    def __init__(self, id, depth, name, cut=None):
        self.id = id
        self.depth = depth
        self.name = name
        self.cut = cut
        self.children = []

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter > 1:
            raise StopIteration
        else:
            self.counter += 1
            return self.next[self.counter-1]

    def __str__(self):
        return "id: {}\nname: {}\ncut: {}\ndepth: {}\nchildren: {}\n".format(self.id, self.name, self.cut, self.depth, self.children)


class Tree():

    def __init__(self, nodes):
        self.nodes = []
        self.nodes_dict = {}
        self.nodes_per_depth = defaultdict(list)
        self.max_depth = 0
        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        if not isinstance(node, Node):
            raise TypeError("Must be Node object")
        self.nodes.append(node)
        self.nodes_dict[node.id] = node
        self.nodes_per_depth[node.depth].append(node.id)
        if node.name == "leaf":
            node.children = [-1, -1]
        if node.depth > self.max_depth:
            self.max_depth = node.depth

    def get_nodes(self):
        return self.nodes

    def get_single_node(self, id):
        return self.nodes_dict[id]

    def get_nodes_in_depth(self, depth):
        return self.nodes_per_depth[depth]

    def add_child_to_node(self, parent_id, child_id):
        self.get_single_node(parent_id).children.append(child_id)
        self.get_single_node(child_id).parent = parent_id

    def get_patterns(self):
        # check if pattern already exists
        try:
            return self.patterns
        except AttributeError as e:
            pass

        # Branches of a (XGB) tree can be of different size. This is a problem when one wants to find all possible paths
        # to the bottom of the tree. The following NULL node was introduced to solve this problem. Every leaf without a
        # children can reference this node as its child. This node itself references itself as a child. In the last step
        # these null nodes get omitted
        self.nodes_dict[-1] = Node(id=-1, depth=-1, name="None")
        self.nodes_dict[-1].children = [-1, -1]
        patterns_to_bottom = [[self.get_single_node(0)]] # root node
        for depth in range(self.max_depth-1):
            patterns_to_bottom = [subpattern+[self.get_single_node(child)] for subpattern in patterns_to_bottom for child in subpattern[-1].children]

        # omit NULL nodes
        patterns_to_bottom = [[node for node in subpattern if node.id != -1 and node.name != "leaf"] for subpattern in patterns_to_bottom]

        # Find unique patterns and exclude subpatterns
        temp_placeholder = [[-2]]
        for pattern in patterns_to_bottom:
            if pattern not in temp_placeholder:
                if set(pattern) <= set(temp_placeholder[-1]):
                    pass
                elif set(pattern) >= set(temp_placeholder[-1]):
                    temp_placeholder[-1] = pattern
                else:
                    temp_placeholder.append(pattern)
        del temp_placeholder[0]

        self.patterns = temp_placeholder
        self.nodes_dict.pop(-1)
        return self.patterns

    def get_patterns_of_length(self, length):
        # Test if pattern to the bottom already exist
        try:
            self.patterns
        except AttributeError:
            self.get_patterns()

        # Test if already done
        try:
            return self.pattern_of_length[length]
        except AttributeError:
            self.patterns_of_length = defaultdict(list)
        except KeyError:
            pass

        already_added = []
        for pattern in self.patterns:
            for i in range(len(pattern)-length+1):
                node_ids = [node.id for node in pattern[i:i+length]]
                if node_ids not in already_added:
                    self.patterns_of_length[length].append(pattern[i:i+length])
                    already_added.append([node.id for node in pattern[i:i+length]])

        return self.patterns_of_length[length]


    def get_operation_from_id(self, parent_id, child_id):
        parent = self.nodes_dict[parent_id]
        if child_id == parent.children[0]:
            return "<"
        elif child_id == parent.children[1]:
            return ">"
        else:
            raise("{} not in {} children: {}.".format(child_id, parent_id, parent.children))

    def get_operation_from_node(self, parent, child):
        return self.get_operation_from_id(parent.id, child.id)


    def __str__(self):
        all_nodes = ""
        for node in self.nodes:
            all_nodes += node.__str__() + "\n"
        return all_nodes


class FullXGB():
    """
    XGB is a fast and efficient implementation of Boosted decision trees.
    See here for further information: https://xgboost.readthedocs.io/en/latest/index.html.

    This class is an extension for easier use and especially analysis of the XGBoost algorithm for classification.
    """

    ####
    # Loading dataset
    ####

    def __init__(self, depth, estimator, train_instances, path=None, name=None, **kwargs):
        """

        Parameters
        ----------
        depth : int
            Maximum depth of each tree.
        estimator : int
            Number of successive trees trained.
        train_instances : list or int
        path : str
            Path where the model and pictures can be saved later. This should be done in advance and not after
            training. If the folder or path does not exist, a error is thrown before the process of training.
        name : str
            Identifier for the chosen tree. Used for saving.
        kwargs : int
            Parameters as specified on the official XGBoost documentation page.
        """

        # Check if destination exists
        if path is not None:
            save_folder = "{}/XGB_{}_dep{}_est{}".format(path, name, maxDepth, estimators)
            if os.path.exists(save_folder):
                shutil.rmtree(save_folder)
                # raise FileExistsError("File already exists.")
            self.save_folder = save_folder
            os.mkdir(self.save_folder)

        # Check valid input for instances
        if isinstance(train_instances, int):
            self.proposed_train = [train_instances]
        elif isinstance(train_instances, list):
            self.proposed_train = train_instances
        else:
            raise TypeError("'Instances argument must be int or list.")

        # Define XGB parameters
        parameters = kwargs
        parameters["max_depth"] = depth
        parameters["n_estimators"] = estimators
        self.parameters = parameters

        # Save all important scores later on
        self.scores = defaultdict(list)
        self.weak_successions = {}
        self.strong_successions = {}

        # Select plot settings
        matplotlib.rcParams["axes.titlesize"] = 30
        matplotlib.rcParams["xtick.labelsize"] = 20
        matplotlib.rcParams["ytick.labelsize"] = 20
        matplotlib.rcParams["axes.labelsize"] = 30
        matplotlib.rcParams["font.size"] = 30

    def train(self, train_x, train_y, val_x, val_y):
        """ Call the xgboost.train function.

        Parameters
        ----------
        train_x : pd.DataFrame
            Training dataframe where the columns are the features and the rows the training instances.
        train_y : np.ndarray or list
            Either one-hot encoded numpy array with dimensions (train_x.shape[0], nr_classes) or list with integers
            indicating the class.
        val_x : pd.DataFrame
            Validation dataframe where the columns are the features and the rows the validation instances.
        val_y : np.ndarray or list
            Either one-hot encoded numpy array with dimensions (val_x.shape[0], nr_classes) or list with integers
            indicating the class.

        Returns
        -------
        None
        """
        self.nr_features = train_x.shape[1]
        self.nr_val = val_x.shape[0]

        assert train_x.shape[1] == val_x.shape[1], "Training and testing set have different number of features."

        self.parameters["objective"] = "binary:logistic"
        # self.parameters["num_class"] = len(np.unique(train_y))

        for instance in self.proposed_train:
            if instance > train_x.shape[0]:
                print(
                    "Ran out of input with instances {}. Only {} present. Ignored.".format(instance, train_x.shape[0]))
                continue

            # Initialize model
            print("Instances: ", instance)

            train_x_sub = train_x.iloc[:instance, ]
            train_y_sub = train_y[:instance]

            # fit model no training data
            self.model = XGBClassifier(**self.parameters)
            start = time.clock()
            self.model.fit(train_x_sub, train_y_sub)
            end = time.clock()

            # Evaluation in validation data
            y_pred = self.model.predict(val_x)
            predictions = [round(value) for value in y_pred]

            val_accuracy = accuracy_score(val_y, predictions)

            try:
                val_auc_score = roc_auc_score(val_y, predictions)
            except ValueError:
                val_auc_score = None

            try:
                val_f1 = f1_score(val_y, predictions)
            except ValueError:
                val_f1 = 0

            # Evaluation in train data
            y_pred = self.model.predict(train_x)
            predictions = [round(value) for value in y_pred]

            train_accuracy = accuracy_score(train_y, predictions)

            try:
                train_auc_score = roc_auc_score(train_y, predictions)
            except ValueError:
                train_auc_score = None

            try:
                train_f1 = f1_score(train_y, predictions)
            except ValueError:
                train_f1 = 0

            # Print useful statistics
            print("Validation accuracy: %.2f%%" % (val_accuracy * 100.0))
            print("Train accuracy: %.2f%%" % (train_accuracy * 100.0))
            print("Fit-Time: ", np.round((end - start) / 60, 3), "\n")

            # Save all statistics
            self.scores["val_accuracy"].append(val_accuracy)
            self.scores["val_auc"].append(val_auc_score)
            self.scores["val_f1"].append(val_f1)

            self.scores["train_accuracy"].append(train_accuracy)
            self.scores["train_auc"].append(train_auc_score)
            self.scores["train_f1"].append(train_f1)

            self.scores["train_time"].append(end - start)
            self.scores["train_examples"].append(instance)

        self.scores["fnames"] = train_x.columns.values
        print("Succesfully finished training.")

    def plot_importances(self, limit=20):
        """ Plot feature importances as measured by the f_score (number of cuts) and gain (quality of cut).

        Arguments
        ---------
        limit : int
            Number of features shown in each plot.

        Returns
        -------
        fig : list
            Figure object containing both importance plots below each other.
        """

        titlestring = ("Instances: {}  |  Accuracy: {}% / {}%  |  ROC AUC: {} / {}  |  F1 {} / {}  |  Time: {}s".format(
            self.scores["train_examples"][-1],
            np.round(self.scores["val_accuracy"][-1], 4) * 100, np.round(self.scores["train_accuracy"][-1], 4) * 100,
            np.round(self.scores["val_auc"][-1], 3), np.round(self.scores["train_auc"][-1], 3),
            np.round(self.scores["val_f1"][-1], 3), np.round(self.scores["train_f1"][-1], 3),
            np.round(self.scores["train_time"][-1], 3)
        ))

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(30, 20))

        # Plot feature importance : Gain"
        xgboost.plot_importance(self.model, ax=ax1,
                                importance_type="gain",
                                xlabel="Gain",
                                max_num_features=limit,
                                title=titlestring)

        # Plot feature importance : Frequency
        xgboost.plot_importance(self.model, ax=ax2,
                                importance_type="weight",
                                xlabel="Weight",
                                max_num_features=limit)

        return fig

    def plot_impact_instances(self):
        """ Plot the scores in dependence of the number of instances used.

        Returns
        -------
        fig : list
            Figure objects containing both importance plots below each other.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(30, 20))

        ax1.plot(self.scores["train_examples"], self.scores["val_accuracy"], marker="o", label="Validation")
        ax1.plot(self.scores["train_examples"], self.scores["train_accuracy"], marker="o", linestyle="dashed",
                 label="Train")
        ax1.set_title("Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.grid()
        ax1.legend()

        ax2.plot(self.scores["train_examples"], self.scores["val_auc"], marker="o", label="Validation")
        ax2.plot(self.scores["train_examples"], self.scores["train_auc"], marker="o", linestyle="dashed", label="Train")
        ax2.set_title("ROC AUC")
        ax2.set_ylabel("ROC AUC")
        ax2.grid()
        ax2.legend()

        ax3.plot(self.scores["train_examples"], self.scores["val_f1"], marker="o", label="Validation")
        ax3.plot(self.scores["train_examples"], self.scores["train_f1"], marker="o", linestyle="dashed", label="Train")
        ax3.set_title("F1 Score")
        ax3.set_ylabel("F1 Score")
        ax3.set_xlabel("#Instances")
        ax3.grid()
        ax3.legend()

        return fig

    def plot_confidence(self, val_x, val_y):
        """ Plot the confidence for predicted and true labels.

        Returns
        -------
        fig : plt.figure
            Figure objects containing both importance plots below each other.
        """
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(30, 20), sharey=True)
        predProba = self.model.predict_proba(val_x)
        fig_true, ax1 = plot_confidence_in_true_label(labels=val_y, predProba=predProba, ax=ax1)
        fig_pred, ax2 = plot_confidence_in_predicted_label(labels=val_y, predProba=predProba, ax=ax2)

        return fig

    def parse_trees(self):
        """ Parse all trees and extract depth, node number, feature and cut value.

        Creates self.trees. This is a list of lists, where every sublist represents a booster (tree) and the
        entries are successive nodes in the tree. The information about the nodes is:
                    (depth, leaf_number, feature, cut value)
        Returns
        -------
        boosters : list
            Same as self.trees. See description above.
        """
        self.model.get_booster().dump_model(self.save_folder + "/xgb_model.txt", with_stats=True)
        split = re.compile('(\s*)([0-9]+):\[([^<>]+)<(-?[0-9]+[\.0-9]*)\]')
        leaf = re.compile('(\s*)([0-9]+):(leaf)')
        self.trees = []

        with open(self.save_folder + '/xgb_model.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if re.search("^booster\[[0-9]+\]", line):  # New booster -> initialize new tree
                    tree = Tree(nodes=[])
                    self.trees.append(tree)

                tree_node = re.findall(split, line)
                if tree_node:
                    tree_node = tree_node[0]
                    id = int(tree_node[1])
                    depth = int(tree_node[0].count("\t"))
                    name = tree_node[2]
                    cut = float(tree_node[3])
                    curr_node = Node(id=id, depth=depth, name=name, cut=cut)
                    tree.add_node(curr_node)

                    if id != 0:
                        parent_id = tree.get_nodes_in_depth(depth-1)[-1]
                        tree.add_child_to_node(parent_id=parent_id, child_id=id)


                tree_leaf = re.findall(leaf, line)
                if tree_leaf:
                    tree_leaf = tree_leaf[0]
                    depth = int(tree_leaf[0].count("\t"))
                    id = int(tree_leaf[1])
                    name = tree_leaf[2]
                    cut = None
                    curr_node = Node(id=id, depth=depth, name=name, cut=cut)
                    tree.add_node(curr_node)
                    parent_id = tree.get_nodes_in_depth(depth - 1)[-1]
                    tree.add_child_to_node(parent_id=parent_id, child_id=id)


    def cuts_hierarchy(self):
        """ Determine in which depth of the tree certain variables were cut.

        Returns
        -------
        cut_hierarchy : pd.DataFrame
            Dictionary where the first row determines the distribution of possible cuts and all others indicate how often
            certain features where cut and in which depth.
        """
        cut_hierachy = pd.DataFrame(
            {"Pos{}".format(depth): np.zeros(len(self.scores["fnames"])) for depth in range(maxDepth)},
            index=self.scores["fnames"])

        for tree in self.trees:
            for node in tree.get_nodes():
                if "leaf" not in node.name:
                    feature = node.name
                    cut_hierachy.loc[feature, "Pos{}".format(node.depth)] += 1

        for col in cut_hierachy.columns:
            cut_hierachy["%{}".format(col)] = np.round(cut_hierachy[col] / sum(cut_hierachy[col]), 2)

        cut_hierachy.sort_values(by=["Pos{}".format(i) for i in range(maxDepth)], ascending=False, inplace=True)
        header = pd.DataFrame(
            {col: ("{} / {}".format(int(sum(cut_hierachy[col])), estimators * 2 ** i) if i < maxDepth else "1") for
             i, col in enumerate(cut_hierachy.columns)}, index=["POSSIBLE"])
        cut_hierachy = header.append(cut_hierachy.astype(str))

        return cut_hierachy


    def find_subgroup_of_cuts(self, search_depth, as_md_table=True):
        """ Find patterns in successive cuts applied to features.

        If variables are cut often successively and pairwise they might identify a subgroup of the population.

        Parameters
        ----------
        depth : int
            Determines the number of subsequent cuts to be analized.

        Returns
        -------
        all_tree_chains : list
            Each element of this list is again a list representing a booster. The elements of these lists are tuples
            of successive cuts.
        OR
        table : str
            Markdown table of results for better representation
        """
        print("Searching for subgroups of depth {}.\n".format(search_depth))

        node_patterns = []
        weak_succession = []
        strong_succession = []
        for t, tree in enumerate(self.trees):  # Go through all trees / boosters
            print("Searching in tree {}...".format(t))

            patterns_of_length_searchdepth = tree.get_patterns_of_length(length=search_depth)

            weak_succession += self.weak_succession(patterns_of_length_searchdepth)
            strong_succession += self.strong_succession(patterns_of_length_searchdepth, tree)
            node_patterns.extend(patterns_of_length_searchdepth)

        self.weak_successions[search_depth] = Counter(weak_succession).most_common()
        self.strong_successions[search_depth] = Counter(strong_succession).most_common()
        if as_md_table:
            table = self.create_md_table(self.weak_successions[search_depth], search_depth) + "\n\n"
            table += self.create_md_table(self.strong_successions[search_depth], search_depth) + "\n\n"
            return table
        else:
            return node_patterns

    def weak_succession(self, succeeding_nodes):
        """ Counts how often two features are cut one after the other regardless of the cut value.
        The order of cuts does matter. A -> B != B -> A
        """
        succeding_names = [" | - | ".join([node.name for node in nodes]) for nodes in succeeding_nodes]
        return succeding_names

    def strong_succession(self, succeeding_nodes, tree):
        """ Counts how often two features are cut one after the other with the same cut value.
        The order of cuts matters. A -> B != B -> A
        """
        operations = [[tree.get_operation_from_node(node1, node2) for node1, node2 in zip(nodes[:-1], nodes[1:])] for nodes in succeeding_nodes]
        operations = [op+["=="] for op in operations]
        succeding_names = [" | - | ".join(["{} {} {}".format(node.name, op, node.cut) for op, node in zip(operation, nodes)]) for operation, nodes in zip(operations, succeeding_nodes)]
        return succeding_names

    def create_md_table(self, succeding_nodes, depth):
        """ Create a markdown table from the successive node patterns.

        Parameters
        ----------
        succeding_nodes : list
            List of strings. Each string contains the the names of features separated by ' | - | ' as constructed by the
            weak_succession & strong_succession subroutines of the method self.find_subgroup_of_cuts(...).
        depth : int
            Length of chain to identify patterns.

        Returns
        -------
        table : str
            Table in markdown format containing information about the cut patterns.
        """
        tableheader = "".join(["| {:<30} ".format("Feature {}".format(i + 1)) for i in range(depth)])
        tableheader += "| {:<10} |\n".format("Count")

        tableheader += "".join(["| {} ".format("-" * 30) for _ in range(depth)])
        tableheader += "| {} |\n".format("-" * 10)

        table_body = ""
        for successive_nodes in succeding_nodes:
            if successive_nodes[1] > 1:
                nodes = successive_nodes[0].split(" | - | ")
                value = str(successive_nodes[1])
                table_body += "".join(["| {:<30} ".format(node) for node in nodes])
                table_body += "| {:<10} |\n".format(value)

        table = tableheader + table_body
        return table

    def cuts_per_feature(self, md_table=True):
        """ Determine the number of times a feature was cut at a certain value.

        Returns
        -------
        table : str
            String in markdown format with inforamtion about feature cuts.
        """
        cut_monitor = defaultdict(list)
        for tree in self.trees:
            for node in tree.nodes:
                if "leaf" not in node.name:
                    feature = node.name
                    cut = node.cut
                    cut_monitor[feature].append(cut)

        for feature in cut_monitor:
            cut_monitor[feature] = Counter(cut_monitor[feature]).most_common()

        if md_table:
            table = "| {:<30} | {:<15} | {:<10} |\n".format("Feature", "Value", "#Cuts")
            table += "| {} | {} | {} |\n".format("-" * 30, "-" * 15, "-" * 10)
            for feature in cut_monitor:
                table += "| {:<30} | {:<15} | {:<10} |\n".format(feature, "",
                                                                 sum([value[1] for value in cut_monitor[feature]]))
                table += "".join(
                    ["| {:<30} | {:<15} | {:<10} |\n".format("", cuts[0], cuts[1]) for cuts in cut_monitor[feature]])

            return table
        else:
            return cut_monitor

    def feature_wise_analysis(self, train_x, train_y, val_x, val_y, max_features=10):
        """ Does a cumulative fit to the subset of 'max_features' best features.

        First a fit is performed only using the best feature, the the best two features, then three,...

        Arguments
        ---------
        train_x : pd.DataFrame
            Training dataframe where the columns are the features and the rows the training instances.
        train_y : np.ndarray or list
            Either one-hot encoded numpy array with dimensions (train_x.shape[0], nr_classes) or list with integers
            indicating the class.
        val_x : pd.DataFrame
            Validation dataframe where the columns are the features and the rows the validation instances.
        val_y : np.ndarray or list
            Either one-hot encoded numpy array with dimensions (val_x.shape[0], nr_classes) or list with integers
            indicating the class.
        max_features : int [10]
            Maximal number of features used for the fit.

        Returns
        -------
        importance_text : str
            String in markdown table format containing the accuracy, roc auc scroe and f1 score per fit.
        """
        importances = self.model.get_booster().get_score(importance_type="gain")
        important_names = []
        important_values = []
        for key, value in importances.items():
            important_names.append(key)
            important_values.append(value)
        sorted_idx = np.argsort(important_values)[::-1]
        important_names = np.array(important_names)[sorted_idx]

        max_thresh = max_features + 1 if len(important_names) > max_features + 1 else len(important_names) + 1

        # Tableheader
        importance_text = "| {:<50} | {:<8} | {:<8} | {:<8} |\n".format("Used features (Cumulative)", "Accuracy",
                                                                        "ROC AUC", "F1-Score")
        importance_text += "| {} | {} | {} | {} |\n".format("-" * 50, "-" * 8, "-" * 8, "-" * 8)
        for thresh in range(1, max_thresh):
            print("Fitting featurewise", thresh, "of", max_thresh - 1)

            # select features using threshold
            select_x_train = train_x[important_names[:thresh]]
            # train model
            selection_model = XGBClassifier(max_depth=maxDepth, n_estimators=estimators, silent=True,
                                            objective="binary:logistic")
            selection_model.fit(select_x_train, train_y)

            # eval model
            select_x_test = val_x[important_names[:thresh]]
            y_pred = selection_model.predict(select_x_test)
            predictions = [np.round(value) for value in y_pred]
            accuracy = accuracy_score(val_y, predictions)
            auc = np.round(roc_auc_score(val_y, predictions), 3)
            f1 = np.round(f1_score(val_y, predictions), 3)
            importance_text += "| {:<50} | {:<8} | {:<8} | {:<8} |\n".format(
                important_names[:thresh][-1],
                round(accuracy * 100.0, 1),
                round(auc, 3), round(f1, 3))

        return importance_text

    def plot_trees(self, step):
        """ Plot the trees via an internal function

        Arguments
        ---------
        step : int
            Number of steps between two plotted trees.

        Returns
        -------
        fig_trees : list
            List of figure objects containing the graphs of the trees.
        """
        fig_trees = []
        for i in range(len(self.trees)):
            if i % step == 0:
                print("Plotting tree {} / {}...".format(i, len(self.trees)))
                try:
                    xgboost.plot_tree(self.model, num_trees=i)
                    fig = matplotlib.pyplot.gcf()
                    fig.set_size_inches(50, 25)
                    ax = plt.gca()
                    ax.set_title("Tree {}".format(i))

                    fig_trees.append(fig)
                except ValueError:
                    print("Bad tree {}".format(i))

        return fig_trees

    def plot_subgroups(self, val_x, val_y, search_depth):
        """

        Parameters
        ----------
        val_x : pd.DataFrame
            Validation dataframe where the columns are the features and the rows the validation instances.
        val_y : np.ndarray or list
            Either one-hot encoded numpy array with dimensions (val_x.shape[0], nr_classes) or list with integers
            indicating the class.
        search_depth : int
            Determines the number of subsequent cuts to be analized.

        Returns
        -------
        """
        figs = []

        hierarchy = self.cuts_hierarchy()
        most_important_features = hierarchy.index[1:6].values
        for feature in most_important_features:
            figs.append(self.D1Hist(val_x, val_y, feature))


        # Patterns in 2D
        weak_successions = self.weak_successions[2]
        for succession in weak_successions[:5]:
            features = succession[0].split(" | - | ")

            figs.append(self.D2Hist(val_x, val_y, features))

        # patterns in >2D
        figs.extend(self.D3Hist(val_x, val_y, search_depth))

        return figs


    def D1Hist(self, val_x, val_y, feature):
        # Patterns in 1D
        monitor = self.cuts_per_feature(md_table=False)
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
        fig.suptitle(feature)

        min_x, max_x = min(val_x[feature]), max(val_x[feature])
        ax1.hist(val_x.loc[val_y == 0, feature], label="Label 0", histtype="stepfilled", bins=20, alpha=0.3,
                 range=(min_x, max_x))
        ax1.hist(val_x.loc[val_y == 1, feature], label="Label 1", histtype="stepfilled", bins=20, alpha=0.3,
                 range=(min_x, max_x))

        important_cuts = [float(cut[0]) for cut in monitor[feature]][:3]
        for i, cut in enumerate(important_cuts):
            if i == 0:
                ax1.axvline(x=cut, color="r", linewidth=5, label="Cuts")
            else:
                ax1.axvline(x=cut, color="r", linewidth=5 / (i + 1))
            plt.text(cut, ax1.get_ylim()[1], str(round(cut, 2)), rotation=0, color="r")
        ax1.set_xlabel(feature)
        ax1.set_ylabel("Count")
        ax1.legend()

        return fig

    def D2Hist(self, val_x, val_y, features):
        temp_data_0 = val_x.loc[val_y == 0, features]
        temp_data_1 = val_x.loc[val_y == 1, features]

        # the scatter plot:
        x0 = temp_data_0.iloc[:, 0].values
        x1 = temp_data_1.iloc[:, 0].values
        y0 = temp_data_0.iloc[:, 1].values
        y1 = temp_data_1.iloc[:, 1].values

        fig, axScatter = plt.subplots(figsize=(30, 20))
        axScatter.scatter(x0, y0, color="blue", marker="_", alpha=0.4, s=50, label="Label 0")
        axScatter.scatter(x1, y1, color="red", marker="P", alpha=0.4, s=50, label="Label 1")
        axScatter.set_xlabel(features[0])
        axScatter.set_ylabel(features[1])
        axScatter.legend()
        axScatter.set_aspect("equal")

        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(axScatter)
        axHistx = divider.append_axes("top", 2., pad=0.1, sharex=axScatter)
        axHisty = divider.append_axes("right", 2., pad=0.1, sharey=axScatter)

        # make some labels invisible
        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHisty.yaxis.set_tick_params(labelleft=False)

        axHistx.hist(x0, color="blue", histtype="stepfilled", bins=30, alpha=0.3)
        axHistx.hist(x1, color="red", histtype="stepfilled", bins=30, alpha=0.3)
        axHisty.hist(y0, orientation='horizontal', color="blue", histtype="stepfilled", bins=30, alpha=0.3)
        axHisty.hist(y1, orientation='horizontal', color="red", histtype="stepfilled", bins=30, alpha=0.3)
        plt.close()

        return fig

    def D3Hist(self, val_x, val_y, search_depth):
        figs = []
        try:
            strong_successions = self.strong_successions[search_depth]
        except KeyError:
            self.find_subgroup_of_cuts(search_depth=search_depth)
            strong_successions = self.strong_successions[search_depth]

        cut_partitioning = {}
        temp_figs = []
        for succession in strong_successions[:5]:
            features = succession[0].split(" | - | ")
            operations = []
            features_cuts = []
            feature_names = []
            cuts = []

            for i, f in enumerate(features):
                if " < " in f:
                    operations.append("<")
                    features_cuts.append(f.split(" < "))
                if " > " in f:
                    operations.append(">=")
                    features_cuts.append(f.split(" > "))
                if " == " in f:
                    operations.append("=")
                    features_cuts.append(f.split(" == "))

                feature_names.append(features_cuts[-1][0])
                cuts.append(float(features_cuts[-1][1]))


            temp_x = val_x.copy()
            temp_x["Label"] = val_y
            for operation, feature_cut in zip(operations[:-1], features_cuts[:-1]):
                if operation == "<":
                    temp_x = temp_x.loc[temp_x[feature_cut[0]] < float(feature_cut[1]), :]
                elif operation == ">=":
                    temp_x = temp_x.loc[temp_x[feature_cut[0]] >= float(feature_cut[1]), :]

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
            fig.suptitle(succession)

            ax.hist(temp_x.loc[temp_x["Label"].values == 0, features_cuts[-1][0]], label="Label 0",
                    histtype="stepfilled", bins=20, alpha=0.3)
            ax.hist(temp_x.loc[temp_x["Label"].values == 1, features_cuts[-1][0]], label="Label 1",
                    histtype="stepfilled", bins=20, alpha=0.3)
            ax.axvline(x=float(features_cuts[-1][1]), color="y")
            ax.set_xlabel(features_cuts[-1][0])
            ax.set_ylabel("Count")
            ax.legend()
            temp_figs.append(fig)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
            temp_X_smaller = temp_x.loc[temp_x[features_cuts[-1][0]] < float(features_cuts[-1][1]), :]
            temp_Y_smaller = temp_X_smaller["Label"]
            temp_X_smaller.drop("Label", axis=1, inplace=True)
            predProba = self.model.predict_proba(temp_X_smaller)

            try:
                ax1 = plot_confidence_in_predicted_label(temp_Y_smaller, predProba, ax=ax1)
            except ValueError:
                pass
            try:
                ax3 = plot_confidence_in_true_label(temp_Y_smaller, predProba, ax=ax3)
            except ValueError:
                pass

            temp_X_greater = temp_x.loc[temp_x[features_cuts[-1][0]] >= float(features_cuts[-1][1]), :]
            temp_Y_greater = temp_X_greater["Label"]
            temp_X_greater.drop("Label", axis=1, inplace=True)
            predProba = self.model.predict_proba(temp_X_greater)
            try:
                ax2 = plot_confidence_in_predicted_label(temp_Y_greater, predProba, ax=ax2)
            except ValueError:
                pass
            try:
                ax4 = plot_confidence_in_true_label(temp_Y_greater, predProba, ax=ax4)
            except ValueError:
                pass

            temp_figs.append(fig)

            title = str(features[:-1]) + "\nLeft: {}".format(features[-1]).replace("==", "<")
            fig.suptitle(title)

            dec = sum(temp_Y_smaller == 0)
            inc = sum(temp_Y_smaller == 1)
            cut_partitioning[succession] = (dec, inc, sum(temp_x["Label"] == 0) - dec, sum(temp_x["Label"]) - inc)

        # Summary plot
        bars_0 = []
        bars_1 = []
        bars_2 = []
        bars_3 = []
        keys = []
        title = ""
        for i, (key, value) in enumerate(cut_partitioning.items()):
            keys.append(key)
            bars_0.append(value[0])
            bars_1.append(value[1])
            bars_2.append(value[2])
            bars_3.append(value[3])
            title += "{}: {}\n".format(i, key)

        barWidth = 0.25
        r0 = np.arange(len(bars_0)) * 3
        r1 = [x + barWidth for x in r0]
        r2 = [x + barWidth * 2 for x in r1]
        r3 = [x + barWidth for x in r2]

        fig = plt.figure(figsize=(30, 20))
        plt.bar(r0, bars_0, color='#888ff7', width=barWidth, edgecolor='white', label='Label 0 <')
        plt.bar(r1, bars_1, color='#f3bbbb', width=barWidth, edgecolor='white', label='Label 1 <')
        plt.bar(r2, bars_2, color='#000ee8', width=barWidth, edgecolor='white', label='Label 0 >')
        plt.bar(r3, bars_3, color='#950000', width=barWidth, edgecolor='white', label='Label 1 >')

        plt.xlabel('Cut', fontweight='bold')
        plt.xticks([r + barWidth * 2 for r in np.arange(len(bars_0)) * 3], np.arange(len(bars_0)))
        plt.legend()
        plt.title(title, fontdict={"size": 25})

        figs.append(fig)
        figs.extend(temp_figs)
        return figs

    def save_model(self):
        """ Saves the latest model as pickle and in xgb intern format.
        """

        with open("{}/Full_XGB.pickle".format(self.save_folder), "wb") as f:
            pickle.dump(self, f)

    def full_analysis(self, train_x, train_y, val_x, val_y, importance_limit=10, tree_plot_step=1, save_model=False):
        """ Wrapper around all functionalities of this class intended for simplified use by the user.

        Parameters
        ----------
        train_x : pd.DataFrame
            Training dataframe where the columns are the features and the rows the training instances.
        train_y : np.ndarray or list
            Either one-hot encoded numpy array with dimensions (train_x.shape[0], nr_classes) or list with integers
            indicating the class.
        val_x : pd.DataFrame
            Validation dataframe where the columns are the features and the rows the validation instances.
        val_y : np.ndarray or list
            Either one-hot encoded numpy array with dimensions (val_x.shape[0], nr_classes) or list with integers
            indicating the class.
        importance_limit : int
            Number of features shown in each plot.
        tree_plot_step : int [1]
            Number of steps between two plotted trees.
        save_model : bool [False]
            If true the whole object gets stored in the same folder as all other data.

        Returns
        -------
        None; everything is saved in the save_folder.
        """
        train_y = np.array(train_y)
        val_y = np.array(val_y)

        try:
            train_y = train_y.argmax(axis=1)
        except ValueError:
            pass

        try:
            val_y = val_y.argmax(axis=1)
        except ValueError:
            pass

        # Train model

        self.train(train_x, train_y, val_x, val_y)
        # Save importance plot
        fig = self.plot_importances(limit=importance_limit)
        fig.savefig("{}/Importances.png".format(self.save_folder))

        # Save progress with used instances
        if len(self.scores["train_examples"]) > 1:
            fig = self.plot_impact_instances()
            fig.savefig("{}/Instances.png".format(self.save_folder))

        # Save confidence plots
        fig = self.plot_confidence(val_x=val_x, val_y=val_y)
        fig.savefig("{}/Confidence.pdf".format(self.save_folder))

        # Parse tree and get information on cuts
        self.parse_trees()

        # Save hierarchy of the cuts
        hierarchy = self.cuts_hierarchy()
        hierarchy.to_csv(self.save_folder + "/cut_hierachy.csv")

        # Save detailed information about used cuts and subgroups
        print("Constructing markdown tables.")
        md_table = "# XGBoost full analysis of cuts\n"
        md_table += "Contains the following information:\n"
        md_table += "- [Cumulative importance](#cumulative-importance)\n"
        md_table += "- [Exact cuts per feature](#exact-cuts)\n"
        md_table += "- [2 successive cuts](#2-cuts)\n"
        md_table += "- [3 successive cuts](#3-cuts)\n\n\n\n"
        md_table += "#### Cumulative Importance\n"
        md_table += self.feature_wise_analysis(train_x, train_y, val_x, val_y, max_features=importance_limit)
        md_table += "#### Exact cuts\n"
        md_table += self.cuts_per_feature()
        md_table += "#### 2 Cuts\n"
        md_table += self.find_subgroup_of_cuts(search_depth=2)
        md_table += "#### 3 Cuts\n"
        md_table += self.find_subgroup_of_cuts(search_depth=3)
        with open("{}/FeatureImportance.md".format(self.save_folder), "w") as f:
            f.write(md_table)

        # Plot important subgroups
        print("Constructing important subgroup plots")
        figs = self.plot_subgroups(val_x, val_y, search_depth=3)
        if self.parameters["max_depth"] >=4:
            figs.extend(self.D3Hist(val_x, val_y, search_depth=4))
        if self.parameters["max_depth"] >=5:
            figs.extend(self.D3Hist(val_x, val_y, search_depth=5))
        savefig(figs, self.save_folder + "/subgroups.pdf")

        # Plot trees
        if tree_plot_step:
            figs = self.plot_trees(step=tree_plot_step)
            if figs != []:
                savefig(figs, self.save_folder + "/trees.pdf")

        # Save model
        if save_model:
            self.save_model()

        print("Everything saved into {}/.".format(self.save_folder))


if __name__ == "__main__":
    ####
    # Load data
    ####
    datafolder = "/home/tneuer/Algorithmen/0TestData/classification/predictSalary"
    train_examples = 300000

    print("Load training data...")
    train_x = pd.read_csv("{}/Processed_train.csv".format(datafolder))
    train_x = train_x.iloc[:train_examples, :]

    with open("{}/Train_Label.pickle".format(datafolder), "rb") as f:
        train_y = pickle.load(f)
        train_y = train_y[:train_examples, :]

    print("Load validation data...")
    val_x = pd.read_csv("{}/Processed_test.csv".format(datafolder))

    with open("{}/Test_Label.pickle".format(datafolder), "rb") as f:
        val_y = pickle.load(f)

    ####
    # Define parameters
    ####

    maxDepth = 10
    estimators = 100
    instances = [50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 200000, 250000, 300000]
    # instances = [150000]

    ####
    # Run algorithm
    ####

    full_xgb = FullXGB(depth=maxDepth, estimator=estimators, path=".", name="PredictSalary", train_instances=instances)
    full_xgb.full_analysis(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, importance_limit=10, tree_plot_step=100)