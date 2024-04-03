import pandas as pd
import numpy as np
import os
import re
import io
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.utils import class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
import pickle
import warnings


def draw_confusion_matrix(y_test, y_pred, model):
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def draw_roc_curve(y_test, y_pred_prob, model):
    # Calcul des taux de faux positifs (FPR) et de vrais positifs (TPR)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    
    # Calcul de l'aire sous la courbe ROC (AUC)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize = (14, 8))
    plt.plot(fpr, tpr, label="ROC curve(area = %0.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="red",label="Random Baseline", linestyle="--")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", size=18)
    plt.legend(loc="lower right")
    plt.show()


def draw_calibration_curve(y_test, y_pred_prob, model):
    frac_pos, mean_pred = calibration_curve(y_test,  y_pred_prob, n_bins=10)

    # Plot the calibration curve
    plt.plot(mean_pred, frac_pos, 's-', label=model)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positive predictions')
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()


def draw_features_importance(pipeline, model, randomF = False):
    if randomF:
        coefficients = pipeline.named_steps[model].feature_importances_
    else: 
        coefficients = pipeline.named_steps[model].coef_[0]
    
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    # Tracer l'importance des caractéristiques
    plt.figure(figsize=(12, 8))
    plt.barh(feature_names, coefficients, color='skyblue')
    plt.xlabel("Features' Importance")
    plt.ylabel('Caractéristiques')
    plt.title("Features' Importance")
    plt.grid(True)
    plt.show()


def draw_prob_distribution(y_pred_prob, model):
    plt.figure(figsize=(10, 8))
    plt.hist(y_pred_prob, bins=10, range=(0, 1), color='blue', alpha=0.7)
    
    plt.xlim(0, 1)
    plt.ylim(0, None)
    
    plt.title('Histogramme des probabilités pour la classe 1')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

