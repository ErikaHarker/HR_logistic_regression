import pandas as pd
import numpy
#libreria para crear los grupos de datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#libreria para el modelo de regresion logistica
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics_lr

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

def class_to_int(df, key):
    #Encuentra las clases de una columna (key) y las ordena alfabeticamente
    array_class = numpy.sort(df[key].unique())
    data_map = {}
    i = 0
    #Asigna desde cero consecutivamente un valor a las clases
    for c in array_class:
        data_map[c] = i
        i += 1
    df[key] = df[key].map(data_map)
    return df


def data_file_clean(data_file):
    #Transformacion de datos
    data_file['Attrition'] = data_file['Attrition'].map({'Yes':1, 'No':0})
    data_file['BusinessTravel'] = data_file['BusinessTravel'].map({'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently': 2})
    data_file = class_to_int(data_file, 'Department')
    data_file = class_to_int(data_file, 'EducationField')
    data_file = class_to_int(data_file, 'Gender')
    data_file = class_to_int(data_file, 'JobRole')
    data_file = class_to_int(data_file, 'MaritalStatus')
    data_file = class_to_int(data_file, 'OverTime')

    #No es una variable porque no cambia
    data_file = class_to_int(data_file, 'Over18')
    return data_file

def logit_regression_sklearn(X, y, X_train, y_train):
                                    
    #Modelo de regresion logistica (sklearn)
    model = LogisticRegression(solver='lbfgs', max_iter=5000)
    model.fit(X_train, y_train)

    print("Información del modelo")
    # Información del modelo
    print("Intercept:", model.intercept_)
    print("Coeficiente:", list(zip(X.columns, model.coef_.flatten(), )))
    print("Accuracy de entrenamiento:", model.score(X, y))

    return model

def logit_regression_statsmodels(X, y, X_train, y_train):
    X_train = sm.add_constant(X_train, prepend=True)
    model = sm.Logit(y_train, X_train).fit(method='bfgs')
    print(model.summary())
    return model
    

data_file_HR = pd.read_csv('hr_employee_attrition.csv', index_col = 0)
data_file_HR = data_file_clean(data_file_HR)

#Definicion de variable objetivo
X = data_file_HR.drop(columns=['Attrition', 'Over18'])
y = data_file_HR['Attrition']

#Data de entrenamiento y prueba
#30% de datos de prueba
X_train, X_test, y_train, y_test = train_test_split(X,  y.values.reshape(-1,1), train_size = 0.7, random_state = 777, shuffle = True)
X_train.to_numpy(dtype='int')

model_sklearn = logit_regression_sklearn(X, y, X_train, y_train)
print("=============================================================")
model_sklearn = logit_regression_statsmodels(X, y, X_train, y_train)




