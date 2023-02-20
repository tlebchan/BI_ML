import numpy as np


def binary_classification_metrics(y_true, y_pred):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    y_true = y_true.astype(int).astype(bool)
    y_pred = y_pred.astype(int).astype(bool)
    
    TP = ((y_true==1)*(y_pred==1)).sum()
    TN = ((y_true==0)*(y_pred==0)).sum()
    FP = ((y_true==0)*(y_pred==1)).sum()
    FN = ((y_true==1)*(y_pred==0)).sum()
    
    if TP+FP > 0:
        precision = TP/(TP+FP)
    else:
        print('ATTENTION, TP+FP IS EQUAL ZERO, precision and f1 is undetermined')
        precision = np.nan
        f1 = np.nan
        
    if TP+FN > 0:
        recall = TP/(TP+FN)
    else:
        print('ATTENTION, TP+FN IS EQUAL ZERO, recall and f1 is undetermined')
        recall = np.nan
        f1 = np.nan
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        print('ATTENTION, precision+recall IS EQUAL ZERO, f1 is undetermined')
        f1 = np.nan
        
    accuracy = (TP+TN)/(TP+FP+TN+FN)
        
    return precision, recall, f1, accuracy

def multiclass_accuracy(y_true, y_pred):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    return (y_true==y_pred).sum()/y_true.shape[0]


def r_squared(y_true, y_pred):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    RSS = ((y_true - y_pred)**2).sum()
    TSS = ((y_true - y_true.mean())**2).sum()
    
    return 1 - RSS/TSS


def mse(y_true, y_pred):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    
    return (((y_true - y_pred)**2).mean())**0.5


def mae(y_true, y_pred):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return (np.abs((y_true - y_pred))).mean()
    