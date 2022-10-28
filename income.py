'''
This file loads the data and generates predictions for the Income Level Prediction problem. It is a binary classification problem of whether a person makes >= 50k annually.

The dataset was extracted by by Barry Becker from the 1994 Census database. There are 14 attributes, including continuous, categorical and integer types. Some attributes may have missing values, recorded as question marks. 

Paper: Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996. (PDF)
'''

import pandas as pd
import pickle
import argparse

# categorical variables
# workclass
# education
# marital-status
# occupation
# relationship
# race
# sex
# native-country

parser = argparse.ArgumentParser(description='A program that evaluates a machine learning model on the Income Prediction dataset.')
parser.add_argument('--train', required=True)
parser.add_argument('--test', required=True)

args = parser.parse_args()
#import numpy as np

#if args.model == 'decisiontree':
#    # lookup the saved model and load it
#    with open('models/rf_clf.model', 'rb') as f:
#        model = pickle.load(f)
#    # Make the predictions on the test test
#    # Generate the output csv
#    pass

def label_dist(data):
    '''
    Find the label distribution.

    input: dataframe
    return:
    '''
    return data['income>50K'].value_counts()


if __name__ == '__main__':

    # Let's compute some statistics about the data
    train_data = pd.read_csv(args.train)

    # How many training instances do we have?
    num_train = len(train_data)
    print("num_train=",num_train)

    # what is the distribution of the labels?
    label_dist(train_data)

    # Compute descriptive statistics for each feature
    for col in train_data.columns:
        print("col=",col)
        stats = train_data[col].describe()
        print(stats)
        print()
   
    # How many and which columns have missing values?
    res = train_data.isin(['?']).sum(axis=0)
    print("res=",res)

    # number of training instances with an unk value
    unk_count_row = train_data.isin(['?']).sum(axis=1)
    print("unk_count_row=",unk_count_row)

    #max number of unk values in a training instance
    unk_max_row = unk_count_row.max()
    print("unk_max_row=",unk_max_row)

    num_max_rows = unk_count_row.isin([3]).sum()
    print("unk_count_row.isin([2]).sum()=",unk_count_row.isin([2]).sum())
    print("unk_count_row.isin([1]).sum()=",unk_count_row.isin([1]).sum())
    print("num_max_rows=",num_max_rows)

    # compute the rest of these later


    # How many rows have missing values?
   #

