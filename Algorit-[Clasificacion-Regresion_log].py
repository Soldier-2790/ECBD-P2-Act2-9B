"""
Algoritmos de clasificación: regresión logística
La regresión logística es un algoritmo de clasificación de aprendizaje supervisado que se utiliza para predecir la probabilidad de una variable objetivo. La naturaleza de la variable objetivo o dependiente es dicotómica, lo que significa que solo habría dos clases posibles.
En palabras simples, la variable dependiente es de naturaleza binaria y tiene datos codificados como 1 (significa éxito / sí) o 0 (significa fracaso / no).
Matemáticamente, un modelo de regresión logística predice P (Y = 1) como una función de X. Es uno de los algoritmos ML más simples que se puede usar para varios problemas de clasificación como detección de spam, predicción de diabetes, detección de cáncer, etc.
--------------------------------
Tipos de regresión logística
En general, la regresión logística significa una regresión logística binaria que tiene variables objetivo binarias, pero puede haber dos categorías más de variables objetivo que pueden predecirse. Según ese número de categorías, la regresión logística se puede dividir en los siguientes tipos:
Binario o Binomial
En tal tipo de clasificación, una variable dependiente tendrá solo dos tipos posibles, 1 y 0. Por ejemplo, estas variables pueden representar éxito o fracaso, sí o no, ganar o perder, etc.
Multinomial
En este tipo de clasificación, la variable dependiente puede tener 3 o más tipos desordenados posibles o los tipos que no tienen importancia cuantitativa. Por ejemplo, estas variables pueden representar "Tipo A" o "Tipo B" o "Tipo C".
Ordinal
En tal tipo de clasificación, la variable dependiente puede tener 3 o más tipos ordenados posibles o los tipos que tienen un significado cuantitativo. Por ejemplo, estas variables pueden representar “pobre” o “bueno”, “muy bueno”, “Excelente” y cada categoría puede tener puntuaciones como 0,1,2,3.
--------------------------------
Supuestos de regresión logística
Antes de sumergirnos en la implementación de la regresión logística, debemos ser conscientes de los siguientes supuestos sobre lo mismo:
En caso de regresión logística binaria, las variables objetivo deben ser siempre binarias y el resultado deseado está representado por el factor nivel 1.
No debe haber multicolinealidad en el modelo, lo que significa que las variables independientes deben ser independientes entre sí.
Debemos incluir variables significativas en nuestro modelo.
Deberíamos elegir un tamaño de muestra grande para la regresión logística.
--------------------------------
Implementación en Python
Ahora implementaremos el concepto anterior de regresión logística binomial en Python. Para este propósito, estamos usando un conjunto de datos de flores multivariante llamado 'iris' que tiene 3 clases de 50 instancias cada una, pero usaremos las dos primeras columnas de características. Cada clase representa un tipo de flor de iris.
Primero, necesitamos importar las bibliotecas necesarias de la siguiente manera:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
A continuación, cargue el conjunto de datos del iris de la siguiente manera:
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
Podemos trazar nuestros datos de entrenamiento a continuación:
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='y', label='1')
plt.legend()
plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='y', label='1')
plt.legend()
plt.show()
