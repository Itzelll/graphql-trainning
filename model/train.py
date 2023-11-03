from os import PathLike
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.impute import SimpleImputer
from joblib import dump
import pandas as pd
import pathlib
import numpy as np
import warnings
warnings.filterwarnings("ignore") 



df = pd.read_csv(pathlib.Path('data/breast_cancer_data.csv'))
#print(df.columns.tolist())
#columns_to_drop = ['diagnosis','1']
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

#X = df
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=10)


print ('Training model.. ')
clf = RandomForestClassifier(n_estimators = 500,
                            max_depth=40
                            )
clf.fit(X_train, y_train)
print ('Saving model..')

dump(clf, pathlib.Path('model/breast_cancer_data-v1.joblib'))
