import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import numpy as np

from utils import *
from operator import itemgetter

import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import datasets
from sklearn import svm
from sklearn.inspection import permutation_importance


from xgboost import XGBRegressor


remove_dolar = lambda x: float(str(x).replace("$", "").replace(",", ""))
    
remove_percent = lambda x: float(str(x).replace('%', ''))

def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df


def split_list_into_columns(df1, col, max_dum = 10):
    '''
    INPUT
        df - pandas dataframe with column cells containing list
        column_name - name of column with list
        max_dummies_num - maximal number of new dummy columns created
    OUTPUT
        df - pandas dataframe with new dummy columns
    '''
    df = copy.deepcopy(df1)
    dist_dict = {}
    
    # Spliting string list into acctual python list
    df[col] = df[col].apply(lambda x: x.replace(']', '').replace("'", '').replace("[", '\
        ').replace('"', '').replace('}', '').replace('{', '').split(',') )
    
    for values_list in df[col]:
        for value in values_list:
            if value in dist_dict:
                dist_dict[value] += 1
            else:
                dist_dict[value] = 1
                
    new_columns = list(dict(sorted(dist_dict.items(), key = itemgetter(1), reverse = True)[:max_dum]).keys())
    
    for new_column in new_columns:
        df[f'{col}_{new_column}'] = df[col].apply(lambda x: 1 if new_column in x else 0)
       
    df = df.drop(columns = [col])
    return df
    
        
def clean_fit_linear_mod(df, response_col, cat_cols, dummy_na, test_size=.3, rand_state=np.random.RandomState()):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column 
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    
    Your function should:
    1. Drop the rows with missing response values
    2. Drop columns with NaN for all the values
    3. Use create_dummy_df to dummy categorical columns
    4. Fill the mean of the column for any missing values 
    5. Split your data into an X matrix and a response vector y
    6. Create training and test sets of data
    7. Instantiate a LinearRegression model with normalized data
    8. Fit your model to the training data
    9. Predict the response for the training data and the test data
    10. Obtain an rsquared value for both the training and test data
    '''
    #Drop the rows with missing response values
    df  = df.dropna(subset = ['price'], axis=0)

    #Drop columns with all NaN values
    df = df.dropna(how='all', axis=1)

    #Dummy categorical variables
    df = create_dummy_df(df, cat_cols, dummy_na)

    # Mean function
    fill_mean = lambda col: col.fillna(col.mode()[0])
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    #Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    #model settings
    model = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=rand_state, n_jobs=-1)

    #Fit
    model.fit(X_train, y_train)
    
    #Predict using your model
    y_test_preds = model.predict(X_test)
    y_train_preds = model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)
    
    features_df = []
    for index, x in enumerate(X_train.columns):
        dir1 = {'column' : x, 'feature_importance': model.feature_importances_[index]}
        features_df.append(dir1)
    features_df = pd.DataFrame(features_df)
    

    return features_df,test_score, train_score, model, X_train, X_test, y_train, y_test