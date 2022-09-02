## When using an experiment to train a model, 
# your script should save the trained model in the outputs folder.
# For example, the following script trains a model using Scikit-Learn, 
# and saves it in the outputs folder using the joblib package

from genericpath import exists
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import os

# get the experiment run context
run = Run.get_context()

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
