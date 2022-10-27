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

def submission_csv(filename, preds):
    '''
    Generate a csv to be submitted on Kaggle.
    '''
    answer = input('Are you sure you want to save a new csv?').lower()
    print("answer=",answer)
    asd
    df = pd.DataFrame(preds, columns=['ID', 'Prediction'])
    df.to_csv(filename+'.csv', index=False)
    print(f'csv written to {filename}.csv.')



if __name__ == '__main__':

    # Let's compute some statistics about the data
    train_data = pd.read_csv(args.train)
    #print("train_data.head()=",train_data.head())
    #counts = train_data.apply(lambda x: len(x.unique()))
    #print("counts=",counts)

#
#    # find most frequent label in the training set.
#    most_freq_label = label_dist(train_data).index[0]
#    print("most_freq_label=",most_freq_label)
#
#    # make preds on test data
#    test_data = pd.read_csv(args.test)
#    preds = [[i,0] for i in test_data['ID'].to_list()] 
#    #FIXME make a submission for the randomforest
#
    

    # make a csv submission for always guessing most freq label.
    #submission_csv('freq-baseline', preds)
