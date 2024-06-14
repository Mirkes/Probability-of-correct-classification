# -*- coding: utf-8 -*-
"""
This software demonstrate non-robustness of Decision threes.
Used database is Breast Cancer database: Obtained from the University Medical
Centre, Institute of Oncology, Ljubljana, Yugoslavia.
Available online: https://archive.ics.uci.edu/dataset/14/breast+cancer



Created on Tue Jun 11 18:39:10 2024


@author: em322
"""
# pylint: disable=C0103

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

def readData():
    '''
    readData read file "breast-cancer.data" from work directory, convert text
    fields into float numbers, removed incomplete records and return result as
    two np.ndarray X for inputs and y for labels.
    This software developed only for this database and is not universal.

    Returns
    -------
    X : 2D np.ndarray
        Contains data matrix with one row for each record
    y : 1D np.ndarray
        Contains labels for records in X

    '''
    # Define lists of categories for decoding
    lists = [["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
             ["ge40", "lt40", "premeno"],
             ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
              "35-39", "40-44", "45-49", "50-54"],
             ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "24-26"],
             ["no", "yes"],
             ["1", "2", "3"],
             ["left", "right"],
             ["left_low", "right_up", "left_up", "right_low", "central"],
             ["no\n", "yes\n"]]
    # Load data and prepare datasets
    with open('breast-cancer.data', encoding="utf-8") as file:
        content = file.readlines()
    # There is no header
    # Create tables for reading
    X = np.zeros((len(content) - 9, 9))
    y = np.zeros(len(content) - 9)
    # p is row to write
    p = 0
    for line in content:
        tmp = line.split(",")
        # Rows with missing value "?" Must be ignored
        if "?" in tmp:
            continue
        # The first is label
        if tmp[0] == "recurrence-events":
            y[p] = 1
        # Decode further attributes
        for kk in range(9):
            X[p, kk] = lists[kk].index(tmp[kk + 1])
        p += 1
    return (X, y)

def splitCreator(X, y, nSplits, testSize):
    '''
    splitCreator creates nSplits independent stratified splits of X and y to training
    and test sets. Results of splitting are written to files "TestTrain_n.npz",
    where n is split number. Each file contains 4 arrays: XTr, XTe, yTr, yTe

    Parameters
    ----------
    X : 2D np.ndarray
        Contains data matrix with one row for each record
    y : 1D np.ndarray
        Contains labels for records in X
    nSplits : int
        number of independent splits.
    testSize : float
        fraction of cases in test size.

    Returns
    -------
    None.

    '''
    for n in range(nSplits):
        XTr, XTe, yTr, yTe = train_test_split(X, y, test_size=testSize,
                                              stratify=y)
        # fil = open("TestTrain_{:03d}.npz".format(n))
        np.savez(f"TestTrain_{n:03d}.npz", XTr=XTr, XTe=XTe,
                 yTr=yTr, yTe=yTe)

def loadSplit(n):
    '''
    load one of previously created splits from file "TestTrain_n.npz",
    where n is split number.

    Parameters
    ----------
    n : int
        the number of the split.

    Returns
    -------
    XTr : 2D np.ndarray
        Inputs of training set.
    XTe : 2D np.ndarray
        Inputs of test set.
    yTr : 1D np.ndarray
        Labels of training set.
    yTe : 1D np.ndarray
        Labels of test set.
    '''
    npzfile = np.load(f"TestTrain_{n:03d}.npz")
    XTr = npzfile['XTr']
    XTe = npzfile['XTe']
    yTr = npzfile['yTr']
    yTe = npzfile['yTe']
    return (XTr, XTe, yTr, yTe)

def oneTree(n):
    '''
    oneTree create Decision tree for dataset in file "TestTrain_n.npz", where n
    is specified splitting. Then tree is tested on test data, basic values (TP,
    TN, FP, FN) are calculated and object with this values is written to file
    "DT_n.pkl".

    Parameters
    ----------
    n : int
        The number of splits to process.

    Returns
    -------
    None.

    '''
    # Load data
    XTr, XTe, yTr, yTe = loadSplit(n)
    # Create tree
    mdl = DecisionTreeClassifier(criterion="entropy").fit(XTr, yTr)
    # predict result for test set
    mdl.pred = mdl.predict(XTe)
    # calculate scores for test set
    a = mdl.pred == 0
    mdl.pure = yTe
    aa = yTe == 0
    mdl.tn = np.sum(a & aa)
    mdl.fn = np.sum(a & ~aa)
    mdl.fp = np.sum(~a & aa)
    mdl.tp = np.sum(~a & ~aa)
    with open(f"DT_{n:03d}.pkl", 'wb') as outp:
        pickle.dump(mdl, outp, pickle.HIGHEST_PROTOCOL)

def reportDT(nSplits):
    '''
    reportDT create report for all trees. For each tree the following values
    are calculated:
    Split number, number of nodes, depth, accuracy, TPR, TNR, PPV, NPV

    Parameters
    ----------
    nSplits : int
        number of splits to process.

    Returns
    -------
    tmp : 2D nd array
        table with 8 columns: Split number, number of nodes, depth, accuracy,
        TPR, TNR, PPV, NPV

    '''

    tmp = np.zeros((nSplits, 8))
    for n in range(nSplits):
        # Load tree
        with open(f"DT_{n:03d}.pkl", 'rb') as inp:
            mdl = pickle.load(inp)
        # Calculate required fields
        tmp[n, 0] = n
        tmp[n, 1] = mdl.tree_.node_count
        tmp[n, 2] = mdl.get_depth()
        tmp[n, 3] = (mdl.tp + mdl.tn) / (mdl.tp + mdl.tn + mdl.fp + mdl.fn)
        tmp[n, 4] = mdl.tp / (mdl.tp + mdl.fn)
        tmp[n, 5] = mdl.tn / (mdl.tn + mdl.fp)
        tmp[n, 6] = mdl.tp / (mdl.tp + mdl.fp)
        tmp[n, 7] = mdl.tn / (mdl.tn + mdl.fn)
    return res

def oneLR(n):
    '''
    oneLR create logistic regression model for dataset in file
    "TestTrain_n.npz", where n is specified splitting. Then model is tested on
    test data, basic values (TP, TN, FP, FN) are calculated and object with
    this values is written to file "LR_n.pkl".

    Parameters
    ----------
    n : int
        The number of splits to process.

    Returns
    -------
    None.

    '''
    # Load data
    XTr, XTe, yTr, yTe = loadSplit(n)
    # Create tree
    mdl = LogisticRegression().fit(XTr, yTr)
    # predict result for test set
    mdl.pred = mdl.predict(XTe)
    # calculate scores for test set
    a = mdl.pred == 0
    mdl.pure = yTe
    aa = yTe == 0
    mdl.tn = np.sum(a & aa)
    mdl.fn = np.sum(a & ~aa)
    mdl.fp = np.sum(~a & aa)
    mdl.tp = np.sum(~a & ~aa)
    with open(f"LR_{n:03d}.pkl", 'wb') as outp:
        pickle.dump(mdl, outp, pickle.HIGHEST_PROTOCOL)

def reportLR(nSplits):
    '''
    reportLR create report for all LR models. For each model the following values
    are calculated:
    Split number, accuracy, TPR, TNR, PPV, NPV

    Parameters
    ----------
    nSplits : int
        number of splits to process.

    Returns
    -------
    tmp : 2D nd array
        table with 6 columns: Split number, accuracy, TPR, TNR, PPV, NPV

    '''

    tmp = np.zeros((nSplits, 6))
    for n in range(nSplits):
        # Load tree
        with open(f"LR_{n:03d}.pkl", 'rb') as inp:
            mdl = pickle.load(inp)
        # Calculate required fields
        tmp[n, 0] = n
        tmp[n, 1] = (mdl.tp + mdl.tn) / (mdl.tp + mdl.tn + mdl.fp + mdl.fn)
        tmp[n, 2] = mdl.tp / (mdl.tp + mdl.fn)
        tmp[n, 3] = mdl.tn / (mdl.tn + mdl.fp)
        tmp[n, 4] = mdl.tp / (mdl.tp + mdl.fp)
        tmp[n, 5] = mdl.tn / (mdl.tn + mdl.fn)
    return res

# What to do. Bit flag:
# 0 (1) - read data and creaate splittings
# 1 (2) - create and test trees, Form preliminary report for formed DTs
# 2 (4) - form LR models and preliminary report.

what = 7
# Number of splits
nSplit = 100
# Fraction of test size
testPart = 0.2

# Load data
if what & 1 == 1:
    (data, labels) = readData()
    # Create splittings
    splitCreator(data, labels, nSplit, testPart)

# Create and test trees
if what & 2 == 2:
    for nn in range(nSplit):
        oneTree(nn)
    # Form preliminary report for formed DTs
    res = reportDT(nSplit)
    with open('DT.csv', 'w', encoding="utf-8") as f:
        f.write("#,# of nodes,Depth,Accuracy,TPR,TNR,PPV,NPV\n")
        for nn in range(nSplit):
            for m in range(3):
                f.write(f"{res[nn, m]:3.0f},")
            for m in range(3, 8):
                f.write(f"{res[nn, m]:6.4f},")
            f.write("\n")

# Generate LR and corresponding report. Draw distributions
if what & 4 == 4:
    for nn in range(nSplit):
        oneLR(nn)
    res = reportLR(nSplit)
    with open('LR.csv', 'w', encoding="utf-8") as f:
        f.write("#,Accuracy,TPR,TNR,PPV,NPV\n")
        for nn in range(nSplit):
            f.write(f"{res[nn, 0]:3.0f}")
            for m in range(1, 6):
                f.write(f",{res[nn, m]:6.4f}")
            f.write("\n")
    for m in range(1,6):
        sns.displot(res[:, m], kind="kde")
        plt.savefig(f"Dist_{m:d}.png")
