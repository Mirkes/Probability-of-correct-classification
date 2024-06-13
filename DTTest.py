# -*- coding: utf-8 -*-
"""
This software demonstrate non-robustness of Decision threes.
Used database is Breast Cancer database: Obtained from the University Medical
Centre, Institute of Oncology, Ljubljana, Yugoslavia. 
Available online: https://archive.ics.uci.edu/dataset/14/breast+cancer



Created on Tue Jun 11 18:39:10 2024


@author: em322
"""

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
    with open('breast-cancer.data') as file:
        content = file.readlines()
    # There is no header
    # Create tables for reading
    X = np.zeros((len(content) - 9, 9))
    y = np.zeros(len(content) - 9)
    # p is row to write
    p = 0
    for k in range(len(content)):
        tmp = content[k].split(",")
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
        np.savez("TestTrain_{:03d}.npz".format(n), XTr=XTr, XTe=XTe, 
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
    npzfile = np.load("TestTrain_{:03d}.npz".format(n))
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
    with open("DT_{:03d}.pkl".format(n), 'wb') as outp:
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
    res : 2D nd array
        table with 8 columns: Split number, number of nodes, depth, accuracy,
        TPR, TNR, PPV, NPV

    '''
    
    res = np.zeros((nSplits, 8))
    for n in range(nSplits):
        # Load tree
        with open("DT_{:03d}.pkl".format(n), 'rb') as inp:
            mdl = pickle.load(inp)        
        # Calculate required fields
        res[n, 0] = n
        res[n, 1] = mdl.tree_.node_count
        res[n, 2] = mdl.get_depth()
        res[n, 3] = (mdl.tp + mdl.tn) / (mdl.tp + mdl.tn + mdl.fp + mdl.fn)
        res[n, 4] = mdl.tp / (mdl.tp + mdl.fn)
        res[n, 5] = mdl.tn / (mdl.tn + mdl.fp)
        res[n, 6] = mdl.tp / (mdl.tp + mdl.fp)
        res[n, 7] = mdl.tn / (mdl.tn + mdl.fn)
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
    with open("LR_{:03d}.pkl".format(n), 'wb') as outp:
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
    res : 2D nd array
        table with 6 columns: Split number, accuracy, TPR, TNR, PPV, NPV

    '''
    
    res = np.zeros((nSplits, 6))
    for n in range(nSplits):
        # Load tree
        with open("LR_{:03d}.pkl".format(n), 'rb') as inp:
            mdl = pickle.load(inp)        
        # Calculate required fields
        res[n, 0] = n
        res[n, 1] = (mdl.tp + mdl.tn) / (mdl.tp + mdl.tn + mdl.fp + mdl.fn)
        res[n, 2] = mdl.tp / (mdl.tp + mdl.fn)
        res[n, 3] = mdl.tn / (mdl.tn + mdl.fp)
        res[n, 4] = mdl.tp / (mdl.tp + mdl.fp)
        res[n, 5] = mdl.tn / (mdl.tn + mdl.fn)
    return res

# What to do. Bit flag:
# 0 (1) - read data
# 1 (2) - creaate splittings
# 2 (4) - create and test trees
# 3 (8) - Form preliminary report for formed DTs
# 4 (16) - form LR models and preliminary report.

what = 0
# Number of splits
nSplits = 100
# Fraction of test size
testSize = 0.2

# Load data
if what & 1 == 1:
    (X, y) = readData()

# Create splittings
if what & 2 == 2:
    splitCreator(X, y, nSplits, testSize)

# Create and test trees
if what & 4 == 4:
    for n in range(nSplits):
        oneTree(n)
        
# Form preliminary report for formed DTs
if what & 8 == 8:
    res = reportDT(nSplits)
    with open('DT.csv', 'w') as f:
        f.write("#,# of nodes,Depth,Accuracy,TPR,TNR,PPV,NPV\n")
        for n in range(nSplits):
            for k in range(3):
                f.write("{:3.0f},".format(res[n, k]))
            for k in range(3, 8):
                f.write("{:6.4f},".format(res[n, k]))
            f.write("\n")

# Generate LR and corresponding report.
if what & 16 == 16:
    for n in range(nSplits):
        oneLR(n)
    res = reportLR(nSplits)
    with open('LR.csv', 'w') as f:
        f.write("#,Accuracy,TPR,TNR,PPV,NPV\n")
        for n in range(nSplits):
            f.write("{:3.0f}".format(res[n, 0]))
            for k in range(1, 6):
                f.write(",{:6.4f}".format(res[n, k]))
            f.write("\n")
# Form distributions    
if 1 == 1:        
    res = reportLR(nSplits)
    for k in range(1,6):
        sns.displot(res[:, k], kind="kde")
        plt.savefig("Dist_{:d}.png".format(k))
