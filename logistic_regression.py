'''
This program trains a logistic regression model.
'''
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import argparse
import pandas as pd
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='A program that evaluates a machine learning model on the Income Prediction dataset.')
parser.add_argument('--train', required=True)
parser.add_argument('--test', required=True)
parser.add_argument('--submission_name')

# Hyperparameters
#parser.add_argument('--num_trees')
#parser.add_argument('--num_trees')

args = parser.parse_args()


def one_hot_encode(df, categorical_vars):
    '''
    one hot encode the categorical variables specified and return a new dataframe.
    '''
    # Our categorical variables are
    # workclass - 
    # education
    # marital-status
    # occupation
    # relationship
    # race
    # sex
    # native-country
    
    # one-hot encoding
    one_hot_df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
    
    # Sanity check on the one-hot encoding
   # # find the value counts for each attribute
   # counts = train_data.apply(lambda x: x.unique())
   # print("counts=",counts)
   # print("type(counts)=",type(counts))
   # print("counts.size=",counts.size)
    #total_columns = sum([len(counts[var]) for var in categorical_vars]) - len(categorical_vars) + (counts.size - len(categorical_vars))
   # print("total_columns=",total_columns)

    return one_hot_df

def submission_csv(filename, preds, ids):
    '''
    Generate a csv to be submitted on Kaggle.
    '''
    data = {'ID' : ids, 'Prediction': preds}
    #answer = input('Are you sure you want to save a new csv?').lower()
    #print("answer=",answer)
    #if answer.lower().strip() == 'y':
    df = pd.DataFrame(data)
    df.to_csv(filename+'.csv', index=False)
    print(f'csv written to {filename}.csv.')

    #else:
        #print('Aborted. Model not saved...')


if __name__ == '__main__':
    # Read data into pandas df
    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)

    # train data has the income>50K column
    print("len(train_data.columns)=",len(train_data.columns))
    # test data has an extra column ID
    print("len(test_data.columns)=",len(test_data.columns))


    # Pre Processing
    # Need to turn the categorical variables into numeric
    # We'll try one-hot encoding them first.

    categorical_vars = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

    # split into train instances and labels
    train_y = train_data['income>50K'].to_numpy()

    # just look at categorical vars in training data
    # remove categorical columns
    # cast to array
    train_data = train_data.drop('income>50K', axis=1)
    numeric_train = train_data.loc[:, ~train_data.columns.isin(categorical_vars)].to_numpy()

    # one hot encode the training data
    #FIXME handle unknown is ignore... how many are being ignored
    # is there an alternate solution?
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    #encoder = OneHotEncoder(handle_unknown='ignore')
    # fit the encoder only on categorical columns of the data
    # transform the subset of the data
    one_hot_train = encoder.fit_transform(train_data[categorical_vars]).toarray()

    # combine the numeric and one-hot features together
    train_X = np.concatenate((numeric_train, one_hot_train), axis=1)
    print("train_X.shape=",train_X.shape)

    # drop ID column of test data 
    test_ids = test_data['ID'].to_numpy()
    test_data = test_data.drop('ID', axis=1)
    numeric_test = test_data.loc[:, ~test_data.columns.isin(categorical_vars)].to_numpy()

    # transform using the same encoding
    one_hot_test = encoder.transform(test_data[categorical_vars]).toarray()

    # combine the numeric and one-hot features together
    test_X = np.concatenate((numeric_test, one_hot_test), axis=1)

    print("test_X.shape=",test_X.shape)

    # Construct the Pipeline
    pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42)
            )

    # Fit the Pipeline
    model = pipe.fit(train_X, train_y)

    acc = accuracy_score(pipe.predict(train_X), train_y)
    print("acc=",acc)

    # Calculate the ROC score for the training data
    print("train_X.shape=",train_X.shape)
    print("train_y.shape=",train_y.shape)
    print("train_y[:10]=",train_y[:10])
    preds_probs = pipe.predict_proba(train_X)[:,1]
    res = roc_auc_score(train_y, preds_probs)
    print("res=",res)

    if args.submission_name is not None:
        asd
        save_dir = f'models/logistic/{args.submission_name}'
        # Save the Pipeline
        with open(f'{save_dir}.model', 'wb') as f:
            pickle.dump(model, f)

        print("pipe.classes_=",pipe.classes_)

        # Make predictions
        preds_probs = pipe.predict_proba(test_X)[:,1]

        # Make Submission CSV
        submission_csv(f'{save_dir}', preds_probs, test_ids)
        

