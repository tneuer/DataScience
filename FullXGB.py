"""
    # Author : Thomas Neuer (tneuer)
    # File Name : DataScience / FullXGB.py
    # Creation Date : 11.11.18 11:27 CET
    # Description : Full XGB analysis developed in EPha.ch. Designed for classification.
"""
#==============================================================================

import os
import re
import time
import pickle
import shutil
import xgboost
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from collections import defaultdict, Counter
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from ModelAnalysis import plot_confidence_in_predicted_label, plot_confidence_in_true_label


def savefig(list_of_plots, path):
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(path)
    for fig in list_of_plots:
        pdf.savefig(fig)
    pdf.close()


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
                print("Ran out of input with instances {}. Only {} present. Ignored.".format(instance, train_x.shape[0]))
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
            print("Fit-Time: ", np.round((end-start)/60, 3), "\n")

            # Save all statistics
            self.scores["val_accuracy"].append(val_accuracy)
            self.scores["val_auc"].append(val_auc_score)
            self.scores["val_f1"].append(val_f1)

            self.scores["train_accuracy"].append(train_accuracy)
            self.scores["train_auc"].append(train_auc_score)
            self.scores["train_f1"].append(train_f1)

            self.scores["train_time"].append(end-start)
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
        ax1.plot(self.scores["train_examples"], self.scores["train_accuracy"], marker="o", linestyle="dashed", label="Train")
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
        self.model.get_booster().dump_model(self.save_folder+"/xgb_model.txt", with_stats=True)
        split = re.compile('(\s*)([0-9]+):\[([^<>]+)<(-?[0-9]+[\.0-9]*)\]')
        boosters = []
        with open(self.save_folder+'/xgb_model.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if re.search("^booster\[[0-9]+\]", line): # New booster -> initialize new list
                    boosters.append([])
                    continue
                tree_node = re.findall(split, line)
                if tree_node:
                    boosters[-1].append(tree_node[0])

        boosters = np.array(boosters)
        self.trees = boosters
        return boosters


    def cuts_hierarchy(self):
        """ Determine in which depth of the tree certain variables were cut.

        Returns
        -------
        cut_hierarchy : pd.DataFrame
            Dictionary where the first row determines the distribution of possible cuts and all others indicate how often
            certain features where cut and in which depth.
        """
        cut_hierachy = pd.DataFrame({"Pos{}".format(depth): np.zeros(len(self.scores["fnames"])) for depth in range(maxDepth)},
                                    index=self.scores["fnames"])

        for tree in self.trees:
            for node in tree:
                feature = node[2]
                depth = node[0].count("\t")
                cut_hierachy.loc[feature, "Pos{}".format(depth)] += 1

        for col in cut_hierachy.columns:
            cut_hierachy["%{}".format(col)] = np.round(cut_hierachy[col] / sum(cut_hierachy[col]), 2)

        cut_hierachy.sort_values(by=["Pos{}".format(i) for i in range(maxDepth)], ascending=False, inplace=True)
        header = pd.DataFrame(
            {col: ("{} / {}".format(int(sum(cut_hierachy[col])), 2*estimators * 2 ** i) if i < maxDepth else "1") for
             i, col in enumerate(cut_hierachy.columns)}, index=["POSSIBLE"])
        cut_hierachy = header.append(cut_hierachy.astype(str))

        return cut_hierachy


    def parse_tree(self, tree):
        """ Parse a single tree and determine which leaves belong to the same branches.

        Arguments
        ---------
        tree : list
            Single parsed booster. Element of self.trees after using object mehtod .parse_trees()

        Returns
        -------
        parsed tree : dict
            Dictionary of a parsed tree where different branches are separated from each other via a "Break".
        """
        try:
            self.trees
        except AttributeError:
            raise AttributeError("No trees attribute. First call self.parse_trees().")

        prev_depth = 0
        parsed_tree = defaultdict(list)
        for node in tree:
            depth = node[0].count("\t")
            if depth < prev_depth: # If going to one leaf above, insert a Break point for all lower nodes.
                for d in range(depth+1, prev_depth+1):
                    parsed_tree[d].append("Break")
            parsed_tree[depth].append(node)
            prev_depth = depth

        for depth in parsed_tree:
            parsed_tree[depth] = parsed_tree[depth] + ["Break"]
        return parsed_tree


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
        def get_deeper_node(parsed_tree, depth, deepest_node):
            # Get next node of the tree
            if depth == deepest_node:
                return parsed_tree[depth].pop(0), 0
            else:
                return parsed_tree[depth].pop(0), 1

        all_tree_chains = []
        for tree in self.trees: # Go through all trees / boosters
            parsed_tree = self.parse_tree(tree) # Parse a single tree

            curr_tree_chains = [] # cut pattern for current tree
            for lowest_node in range(max(parsed_tree.keys())-search_depth+2):
                deepest_node = lowest_node+search_depth-1
                copy_parsed_tree = {i: parsed_tree[i][:] for i in range(lowest_node, deepest_node+1)}
                depth = lowest_node
                chain = [0]*search_depth
                depth_to_index = {d: i for i, d in enumerate(range(lowest_node, deepest_node+1))} #converts depth to list index

                while True:
                    next_node, advance = get_deeper_node(copy_parsed_tree, depth, deepest_node) # get next node

                    chain[depth_to_index[depth]] = next_node # append node to chain, depth ends are marked with "Break"
                    depth += advance
                    isBreak = [True if node is "Break" else False for node in chain] # If break is contained update chain and move one step up
                    if any(isBreak):
                        depth = depth - 1
                        idx = np.where(isBreak)[0][0]
                        chain[idx:] = [0] * (len(chain) - idx)

                    elif all([True if node != 0 else False for node in chain]): # Chain is filled and valid
                        curr_tree_chains.append(chain[:])

                    if copy_parsed_tree[depth] == []: # Can't move up in the tree
                        break

            all_tree_chains.append(curr_tree_chains)

        if as_md_table:
            table = self.create_md_table(self.weak_succession(all_tree_chains), search_depth) + "\n\n"
            table += self.create_md_table(self.strong_succession(all_tree_chains), search_depth) + "\n\n"
            return table
        else:
            return all_tree_chains

    def weak_succession(self, trees_succeeding_nodes):
        """ Counts how often two features are cut one after the other regardless of the cut value.
        The order of cuts does matter. A -> B != B -> A
        """
        succeding_nodes = [l for lists in trees_succeeding_nodes for l in lists]
        succeding_names = [" | - | ".join([node[2] for node in nodes]) for nodes in succeding_nodes]
        return Counter(succeding_names).most_common()


    def strong_succession(self, trees_succeeding_nodes):
        """ Counts how often two features are cut one after the other with the same cut value.
        The order of cuts matters. A -> B != B -> A
        """
        succeding_nodes = [l for lists in trees_succeeding_nodes for l in lists]
        succeding_names = [" | - | ".join([" < ".join([node[2], node[3]]) for node in nodes]) for nodes in succeding_nodes]
        return Counter(succeding_names).most_common()

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
        tableheader = "".join(["| {:<40} ".format("Feature {}".format(i+1)) for i in range(depth)])
        tableheader += "| {:<10} |\n".format("Count")

        tableheader += "".join(["| {} ".format("-"*40) for _ in range(depth)])
        tableheader += "| {} |\n".format("-"*10)

        table_body = ""
        for successive_nodes in succeding_nodes:
            nodes = successive_nodes[0].split(" | - | ")
            value = str(successive_nodes[1])
            table_body += "".join(["| {:<40} ".format(node) for node in nodes])
            table_body += "| {:<10} |\n".format(value)

        table = tableheader + table_body
        return table


    def cuts_per_feature(self):
        """ Determine the number of times a feature was cut at a certain value.

        Returns
        -------
        table : str
            String in markdown format with inforamtion about feature cuts.
        """
        cut_monitor = defaultdict(list)
        for tree in self.trees:
            for node in tree:
                feature = node[2]
                cut = node[3]
                cut_monitor[feature].append(cut)

        for feature in cut_monitor:
            cut_monitor[feature] = Counter(cut_monitor[feature]).most_common()

        table = "| {:<40} | {:<15} | {:<10} |\n".format("Feature", "Value", "#Cuts")
        table += "| {} | {} | {} |\n".format("-" * 40, "-" * 15, "-" * 10)
        for feature in cut_monitor:
            table += "| {:<40} | {:<15} | {:<10} |\n".format(feature, "", sum([value[1] for value in cut_monitor[feature]]))
            table += "".join(
                ["| {:<40} | {:<15} | {:<10} |\n".format("", cuts[0], cuts[1]) for cuts in cut_monitor[feature]])

        return table


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

        max_thresh = max_features+1 if len(important_names) > max_features+1 else len(important_names)

        # Tableheader
        importance_text = "| {:<50} | {:<8} | {:<8} | {:<8} |\n".format("Used features (Cumulative)", "Accuracy", "ROC AUC", "F1-Score")
        importance_text += "| {} | {} | {} | {} |\n".format("-"*50, "-"*8, "-"*8, "-"*8)
        for thresh in range(1, max_thresh):
            print("Fitting featurewise", thresh, "of", max_thresh-1)

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
                xgboost.plot_tree(self.model, num_trees=i)
                fig = matplotlib.pyplot.gcf()
                fig.set_size_inches(50, 25)
                ax = plt.gca()
                ax.set_title("Tree {}".format(i))

                fig_trees.append(fig)

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

        # Patterns in 1D
        hierarchy = self.cuts_hierarchy()
        most_important_features = hierarchy.index[1:6].values
        for feature in most_important_features:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(30, 20))
            fig.suptitle(feature)

            ax1.hist(val_x.loc[val_y==0, feature], label="Label 0", histtype="step")
            ax1.hist(val_x.loc[val_y==1, feature], label="Label 1", histtype="step")
            ax1.set_xlabel(feature)
            ax1.set_ylabel("Count")
            ax1.legend()

            figs.append(fig)

        # Patterns in 2D
        patterns = self.find_subgroup_of_cuts(search_depth=2, as_md_table=False)
        weak_successions = self.weak_succession(patterns)
        for succession in weak_successions[:5]:
            features = succession[0].split(" | - | ")
            temp_data_0 = val_x.loc[val_y==0, features]
            temp_data_1 = val_x.loc[val_y==1, features]

            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(30,20), sharey=True, sharex=True)

            h, _, _, im = ax0.hist2d(temp_data_0.iloc[:, 0], temp_data_0.iloc[:, 1])
            plt.colorbar(im, ax=ax0)
            ax0.set_xlabel(features[0])
            ax0.set_ylabel(features[1])
            ax0.set_title("Label 0")

            h, _, _, im = ax1.hist2d(temp_data_1.iloc[:, 0], temp_data_1.iloc[:, 1])
            plt.colorbar(im, ax=ax1)
            ax1.set_xlabel(features[0])
            ax1.set_ylabel(features[1])
            ax1.set_title("Label 1")

            figs.append(fig)

        # patterns in >2D
        patterns = self.find_subgroup_of_cuts(search_depth=search_depth, as_md_table=False)
        strong_successions = self.strong_succession(patterns)
        cut_partitioning = {}
        temp_figs = []
        for succession in strong_successions[:5]:
            features = succession[0].split(" | - | ")
            features_cuts = [f.split(" < ") for f in features]
            temp_x = val_x.copy()
            temp_x["Label"] = val_y
            for feature_cut in features_cuts:
                temp_x = temp_x.loc[temp_x[feature_cut[0]] < float(feature_cut[1]), :]

            temp_y = temp_x["Label"]
            temp_x.drop("Label", inplace=True, axis=1)

            fig = self.plot_confidence(val_x=temp_x, val_y=temp_y)
            fig.suptitle(succession)
            temp_figs.append(fig)

            dec = sum(temp_y==0)
            inc = sum(temp_y==1)
            cut_partitioning[succession] = (dec, inc, sum(val_y==0)-dec, sum(val_y==1)-inc)

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
        r2 = [x + barWidth*2 for x in r1]
        r3 = [x + barWidth for x in r2]

        fig = plt.figure(figsize=(30,20))
        plt.bar(r0, bars_0, color='#888ff7', width=barWidth, edgecolor='white', label='Label 0 <')
        plt.bar(r1, bars_1, color='#f3bbbb', width=barWidth, edgecolor='white', label='Label 1 <')
        plt.bar(r2, bars_2, color='#000ee8', width=barWidth, edgecolor='white', label='Label 0 >')
        plt.bar(r3, bars_3, color='#950000', width=barWidth, edgecolor='white', label='Label 1 >')

        plt.xlabel('Cut', fontweight='bold')
        plt.xticks([r + barWidth*2 for r in np.arange(len(bars_0))*3], np.arange(len(bars_0)))
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
        hierarchy.to_csv(self.save_folder+"/cut_hierachy.csv")

        # Save detailed information about used cuts and subgroups
        md_table = "# XGBoost full analysis of cuts\n"
        md_table += "Contains the following information:\n"
        md_table += "- [Cumulative importance](#cumulative-importance)\n"
        md_table += "- [Exact cuts per feature](#exact-cuts)\n"
        md_table += "- [2 successive cuts](#2-cuts)\n"
        md_table += "- [3 successive cuts](#3-cuts)\n\n\n\n"
        md_table += "#### Cumulative Importance\n"
        md_table += self.feature_wise_analysis(train_x, train_y, val_x, val_y)
        md_table += "#### Exact cuts\n"
        md_table += self.cuts_per_feature()
        md_table += "#### 2 Cuts\n"
        md_table += self.find_subgroup_of_cuts(search_depth=2)
        md_table += "#### 3 Cuts\n"
        md_table += self.find_subgroup_of_cuts(search_depth=3)
        with open("{}/FeatureImportance.md".format(self.save_folder), "w") as f:
            f.write(md_table)

        # Plot trees
        if tree_plot_step:
            figs = self.plot_trees(step=tree_plot_step)
            savefig(figs, self.save_folder + "/trees.pdf")

        # Plot important subgroups
        figs = self.plot_subgroups(val_x, val_y, search_depth=3)
        savefig(figs, self.save_folder + "/subgroups.pdf")

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

    maxDepth = 3
    estimators = 5
    instances = [50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 200000, 250000, 300000]
    # instances = [150000]

    ####
    # Run algorithm
    ####

    full_xgb = FullXGB(depth=maxDepth, estimator=estimators, path=".", name="PredictSalary", train_instances=instances)
    full_xgb.full_analysis(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, importance_limit=20, tree_plot_step=20)
