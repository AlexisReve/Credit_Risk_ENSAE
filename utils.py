import pandas as pd
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
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
    
    plt.figure(figsize = (10, 6))
    plt.plot(fpr, tpr, label="ROC curve(area = %0.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="red",label="Random Baseline", linestyle="--")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve : {}".format(model), size=18)
    plt.legend(loc="lower right")
    plt.show()


def draw_calibration_curve(y_test, y_pred_prob, model):
    frac_pos, mean_pred = calibration_curve(y_test,  y_pred_prob, n_bins=10)

    # Plot the calibration curve
    plt.figure(figsize = (10, 6))
    plt.plot(mean_pred, frac_pos, 's-', label=model)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positive predictions')
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()


def draw_prob_distribution(y_pred_prob, model):
    plt.figure(figsize = (10, 6))
    plt.hist(y_pred_prob, bins=10, range=(0, 1), color='blue', alpha=0.7)
    
    plt.xlim(0, 1)
    plt.ylim(0, None)
    
    plt.title('Histogramme des probabilités pour la classe 1')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def top_features(features_importance , pipeline, model, randomF = False):
    if randomF:
        coefficients = pipeline.named_steps[model].feature_importances_
    else:
        coefficients = pipeline.named_steps[model].coef_[0]
    
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False)['Feature'][:5]
    features_importance[model] = top_features
    return features_importance


def pipeline_logreg_benchmark(X_train, y_train, X_test, y_test, model_result, features_importance):
    pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('LogisticRegression_Benchmark', LogisticRegression(solver='saga', class_weight = weight_dict,
                                  max_iter=5000, n_jobs=-1))  
])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 1)
    model = "LogisticRegression_Benchmark"
    result = pd.DataFrame({"Model" : [model],
                       "Accuracy" : [accuracy_score(y_test, y_pred)],
                       "Recall" : [recall_score(y_test, y_pred)],
                       "F1_score" : [f1_score(y_test, y_pred, average="macro")],
                       "AUC" : [auc(fpr, tpr)]}
                       )
    model_result = pd.concat([model_result, result])
    features_importance = top_features(features_importance , pipeline, model, randomF = False)
    draw_confusion_matrix(y_test, y_pred, model)
    draw_roc_curve(y_test, y_pred_prob, model)
    draw_prob_distribution(y_pred_prob, model)
    draw_calibration_curve(y_test, y_pred_prob, model)
    return model_result, features_importance


def pipeline_logreg_cv(X_train, y_train, X_test, y_test, model_result, features_importance):
    #y_train = y_train['SeriousDlqin2yrs']
    param_grid = {'LogisticRegression_cv__C': np.logspace(-10, 6, 17, base=2),
              'LogisticRegression_cv__penalty': ['l1', 'l2'],
               'LogisticRegression_cv__class_weight': ['balanced', weight_dict]} 
    pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('LogisticRegression_cv', LogisticRegression(solver='saga', max_iter=5000))  
])
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', error_score='raise',
                          n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    y_pred_prob = grid_search.predict_proba(X_test)[:, 1]
    best_pipeline = grid_search.best_estimator_
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 1)
    model = "LogisticRegression_cv"
    result = pd.DataFrame({"Model" : [model],
                       "Accuracy" : [accuracy_score(y_test, y_pred)],
                       "Recall" : [recall_score(y_test, y_pred)],
                       "F1_score" : [f1_score(y_test, y_pred, average="macro")],
                       "AUC" : [auc(fpr, tpr)]}
                       )
    model_result = pd.concat([model_result, result])
    
    features_importance = top_features(features_importance , best_pipeline, model, randomF = False)
    draw_confusion_matrix(y_test, y_pred, model)
    draw_roc_curve(y_test, y_pred_prob, model)
    draw_prob_distribution(y_pred_prob, model)
    draw_calibration_curve(y_test, y_pred_prob, model)
    return model_result, features_importance


def pipeline_randomF_benchmark(X_train, y_train, X_test, y_test, model_result, features_importance):
    pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('randomF', RandomForestClassifier(class_weight = weight_dict,
                                  n_jobs=-1))  
])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 1)
    model = "randomF"
    result = pd.DataFrame({"Model" : [model],
                       "Accuracy" : [accuracy_score(y_test, y_pred)],
                       "Recall" : [recall_score(y_test, y_pred)],
                       "F1_score" : [f1_score(y_test, y_pred, average="macro")],
                       "AUC" : [auc(fpr, tpr)]}
                       )
    model_result = pd.concat([model_result, result])
    
    features_importance = top_features(features_importance , pipeline, model, randomF = True)
    draw_confusion_matrix(y_test, y_pred, model)
    draw_roc_curve(y_test, y_pred_prob, model)
    draw_prob_distribution(y_pred_prob, model)
    draw_calibration_curve(y_test, y_pred_prob, model)
    return model_result, features_importance


def pipeline_randomF_cv(X_train, y_train, X_test, y_test, model_result, features_importance):
    param_grid = {
    'randomF_cv__n_estimators': [100, 300],
    'randomF_cv__max_features': ['sqrt', 'log2'],
    'randomF_cv__min_samples_split': [2, 10],
    'randomF_cv__min_samples_leaf': [1, 4],
    'randomF_cv__class_weight': [weight_dict]
}
    pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('randomF_cv', RandomForestClassifier(n_jobs=-1))  
])
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', error_score='raise',
                          n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    y_pred_prob = grid_search.predict_proba(X_test)[:, 1]
    best_pipeline = grid_search.best_estimator_
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 1)
    model = "randomF_cv"
    result = pd.DataFrame({"Model" : [model],
                       "Accuracy" : [accuracy_score(y_test, y_pred)],
                       "Recall" : [recall_score(y_test, y_pred)],
                       "F1_score" : [f1_score(y_test, y_pred, average="macro")],
                       "AUC" : [auc(fpr, tpr)]}
                       )
    model_result = pd.concat([model_result, result])
    
    features_importance = top_features(features_importance , best_pipeline, model, randomF = True)
    draw_confusion_matrix(y_test, y_pred, model)
    draw_roc_curve(y_test, y_pred_prob, model)
    draw_prob_distribution(y_pred_prob, model)
    draw_calibration_curve(y_test, y_pred_prob, model)
    return model_result, features_importance


def pipeline_xgb_benchmark(X_train, y_train, X_test, y_test, model_result, features_importance):
    pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('XGB_Benchmark', XGBClassifier())  
])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 1)
    model = "XGB_Benchmark"
    result = pd.DataFrame({"Model" : [model],
                       "Accuracy" : [accuracy_score(y_test, y_pred)],
                       "Recall" : [recall_score(y_test, y_pred)],
                       "F1_score" : [f1_score(y_test, y_pred, average="macro")],
                       "AUC" : [auc(fpr, tpr)]}
                       )
    model_result = pd.concat([model_result, result])
    
    features_importance = top_features(features_importance , pipeline, model, randomF = True)
    draw_confusion_matrix(y_test, y_pred, model)
    draw_roc_curve(y_test, y_pred_prob, model)
    draw_prob_distribution(y_pred_prob, model)
    draw_calibration_curve(y_test, y_pred_prob, model)
    return model_result, features_importance


def pipeline_xgb_cv(X_train, y_train, X_test, y_test, model_result, features_importance):
    param_grid = {
    'XGB_cv__learning_rate': [0.1, 0.01],
    'XGB_cv__n_estimators': [50, 100, 200],
    'XGB_cv__gamma': [0, 0.1, 0.01],
    'XGB_cv__scale_pos_weight' : [None] + list(weights)
}
    pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('XGB_cv', XGBClassifier(n_jobs=-1))  
])
    grid_search = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', error_score='raise',
                          n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    y_pred_prob = grid_search.predict_proba(X_test)[:, 1]
    best_pipeline = grid_search.best_estimator_
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 1)
    model = "XGB_cv"
    result = pd.DataFrame({"Model" : [model],
                       "Accuracy" : [accuracy_score(y_test, y_pred)],
                       "Recall" : [recall_score(y_test, y_pred)],
                       "F1_score" : [f1_score(y_test, y_pred, average="macro")],
                       "AUC" : [auc(fpr, tpr)]}
                       )
    model_result = pd.concat([model_result, result])
    pickle.dump(best_pipeline, open('model/xgb_model.pkl', 'wb'))
    features_importance = top_features(features_importance , best_pipeline, model, randomF = True)
    draw_confusion_matrix(y_test, y_pred, model)
    draw_roc_curve(y_test, y_pred_prob, model)
    draw_prob_distribution(y_pred_prob, model)
    draw_calibration_curve(y_test, y_pred_prob, model)
    return model_result, features_importance
