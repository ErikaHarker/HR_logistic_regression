import pandas as pd
import numpy
#libreria para crear los grupos de datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#libreria para el modelo de regresion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#Librerias para arboles
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

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

def train_evaluate_sklearn(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("======== Metrics")
    print("Accuracy:", accuracy_score(y_test,y_pred))
    print("Confusion matrix: \n", confusion_matrix(y_test,y_pred))
    return model
    

data_file_HR = pd.read_csv('hr_employee_attrition.csv', index_col = 0)
data_file_HR = data_file_clean(data_file_HR)

#Definicion de variable objetivo
X = data_file_HR.drop(columns=['Attrition', 'Over18'])
y = data_file_HR['Attrition']

#Data de entrenamiento y prueba
#30% de datos de prueba
X_train, X_test, y_train, y_test = train_test_split(X,  y, train_size = 0.7, random_state = 777, shuffle = True)
X_train.to_numpy(dtype='int')

#Pruebas con diferentes modelos de la libreria sklearn

model_logistic = LogisticRegression(solver='lbfgs', max_iter=5000)
model_logistic = train_evaluate_sklearn(model_logistic, X_train, X_test, y_train, y_test)
#Los Coeficiente solo se pueden obtener en algoritmos lineales
print("Coeficiente:", list(zip(X.columns, model_logistic.coef_.flatten(), )))

model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, max_features=4)
model_tree = train_evaluate_sklearn(model_tree, X_train, X_test, y_train, y_test)

model_svm = svm.SVC()
model_svm = train_evaluate_sklearn(model_svm, X_train, X_test, y_train, y_test)







