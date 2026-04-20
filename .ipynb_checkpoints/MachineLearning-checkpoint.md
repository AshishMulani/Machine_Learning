# Imports



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model\_selection import train\_test\_split

from sklearn.linear\_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder, StandarScaler, OneHotEncoder

from sklearn.compose import  ColumnTransformer

from sklearn.metrics import accuracy\_score, confusion\_matrix, f1\_score

from sklearn.metrics import recall\_score, precision\_score, classification\_report, roc\_auc\_score, log\_loss



from sklearn.neighbors import KNeighborsClassifier



import os

os.chdir("D:/Machine\_Learning/Cases") #fetch the dataset from the folder



# LabelEncoder

le = LabelEncoder()

sonar\['Class'] = le.fit\_transform(sonar\['Class'])

X , y = sonar.drop('Class', axis=1), sonar\['Class']



X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size= 0.3, random\_state=26, stratify= sonar\['Class'])





# OneHotEncoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import  ColumnTransformer



from sklearn.compose import make\_column\_selector



ohe=OneHotEncoder(sparse\_output=False,drop="first").set\_output(transform="pandas")



trans = ColumnTransformer(

&#x20;   transformers=\[("OHE", ohe, make\_column\_selector(dtype\_include=object))], remainder="passthrough",

&#x20;   verbose\_feature\_names\_out=False).set\_output(transform="pandas")



X\_trn\_ohe = trans.fit\_transform(X\_train)

X\_tst\_ohe = trans.transform(X\_test)

