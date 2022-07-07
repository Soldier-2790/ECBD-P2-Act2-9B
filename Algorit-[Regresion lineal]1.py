"""
Example 2 - En el siguiente ejemplo de implementación de Python, usamos un conjunto de datos de diabetes de scikit-learn.
Primero, comenzaremos con la importación de paquetes necesarios de la siguiente manera:
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
A continuación, cargaremos el conjunto de datos de diabetes y crearemos su objeto:
diabetes = datasets.load_diabetes()
Mientras implementamos SLR, usaremos solo una característica de la siguiente manera:
X = diabetes.data[:, np.newaxis, 2]
A continuación, necesitamos dividir los datos en conjuntos de entrenamiento y prueba de la siguiente manera:
X_train = X[:-30]
X_test = X[-30:]
A continuación, debemos dividir el objetivo en conjuntos de entrenamiento y prueba de la siguiente manera:
y_train = diabetes.target[:-30]
y_test = diabetes.target[-30:]
Ahora, para entrenar el modelo, necesitamos crear un objeto de regresión lineal de la siguiente manera:
regr = linear_model.LinearRegression()
A continuación, entrene el modelo utilizando los conjuntos de entrenamiento de la siguiente manera:
regr.fit(X_train, y_train)
A continuación, haga predicciones utilizando el conjunto de pruebas de la siguiente manera:
y_pred = regr.predict(X_test)
A continuación, imprimiremos algunos coeficientes como MSE, puntuación de varianza, etc. de la siguiente manera:
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
Ahora, grafique las salidas de la siguiente manera:
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
Salida
Coefficients:
   [941.43097333]
Mean squared error: 3035.06
Variance score: 0.41
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]
X_train = X[:-30]
X_test = X[-30:]
y_train = diabetes.target[:-30]
y_test = diabetes.target[-30:]
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
