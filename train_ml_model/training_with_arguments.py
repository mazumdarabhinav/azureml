## You can increase the flexibility of script-based 
# experiments by using arguments to set variables in the script.
# To use parameters in a script, you must use a library such as 
# argparse to read the arguments passed to the script 
# and assign them to variables. 
# For example, the following script reads an argument named 
# --reg-rate, which is used to set the regularization rate 
# hyperparameter for the logistic regression algorithm 
# used to train a model.

from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import os

# get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument(
    '--reg-rate',
    type=float,
    dest='reg_rate',
    default=0.01
)
args = parser.parse_args()
reg = args.reg_rate

# prepare the dataset
diabetes = datasets.load_diabetes(as_frame=True)
X, y = diabetes["data"], diabetes["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver="liblinear")
model.fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', acc)

# save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

# complete the run
run.complete()
