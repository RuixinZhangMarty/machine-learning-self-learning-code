# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:55:19 2019

@author: z
"""
###From Raw Data to Predictions
%matplotlib inline
from __future__ import division
from __future__ import print_function

# ignore deprecation warnings in sklearn
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)

from data.multilabel import multilabel_sample_dataframe, multilabel_train_test_split
from features.SparseInteractions import SparseInteractions
from models.metrics import multi_multi_log_loss



###Load Data
path_to_training_data = os.path.join(os.pardir,
                                     'data',
                                     'TrainingSet.csv')

df = pd.read_csv(path_to_training_data, index_col=0)

print(df.shape)


###Resample Data
LABELS = ['Function',
          'Use',
          'Sharing',
          'Reporting',
          'Student_Type',
          'Position_Type',
          'Object_Type', 
          'Pre_K',
          'Operating_Status']

NON_LABELS = [c for c in df.columns if c not in LABELS]

SAMPLE_SIZE = 40000

sampling = multilabel_sample_dataframe(df,
                                       pd.get_dummies(df[LABELS]),
                                       size=SAMPLE_SIZE,
                                       min_count=25,
                                       seed=43)

dummy_labels = pd.get_dummies(sampling[LABELS])

X_train, X_test, y_train, y_test = multilabel_train_test_split(sampling[NON_LABELS],
                                                               dummy_labels,
                                                               0.2,
                                                               min_count=3,
                                                               seed=43)



###Create preprocessing tools
NUMERIC_COLUMNS = ['FTE', "Total"]

def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ Takes the dataset as read in, drops the non-feature, non-text columns and
        then combines all of the text columns into a single vector that has all of
        the text for a row.
        
        :param data_frame: The data as read in with read_csv (no preprocessing necessary)
        :param to_drop (optional): Removes the numeric and label columns by default.
    """
    # drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    
    # replace nans with blanks
    text_data.fillna("", inplace=True)
    
    # joins all of the text items in a row (axis=1)
    # with a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

from sklearn.preprocessing import FunctionTransformer

get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

get_text_data.fit_transform(sampling.head(5))

get_numeric_data.fit_transform(sampling.head(5))

from sklearn.metrics.scorer import make_scorer

log_loss_scorer = make_scorer(multi_multi_log_loss)



###Train model pipeline
from sklearn.feature_selection import chi2, SelectKBest

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MaxAbsScaler

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

%%time

# set a reasonable number of features before adding interactions
chi_k = 300

# create the pipeline object
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# fit the pipeline to our training data
pl.fit(X_train, y_train.values)

# print the score of our trained pipeline on our test set
print("Logloss score of trained pipeline: ", log_loss_scorer(pl, X_test, y_test.values))



###Predict holdout set and write submission
path_to_holdout_data = os.path.join(os.pardir,
                                    'data',
                                    'TestSet.csv')

# Load holdout data
holdout = pd.read_csv(path_to_holdout_data, index_col=0)

# Make predictions
predictions = pl.predict_proba(holdout)

# Format correctly in new DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)


# Save prediction_df to csv called "predictions.csv"
prediction_df.to_csv("predictions.csv")