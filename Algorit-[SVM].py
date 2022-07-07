"""
Máquina de vectores de soporte (SVM)
Las máquinas de vectores de soporte (SVM) son algoritmos de aprendizaje automático supervisados ​​potentes pero flexibles que se utilizan tanto para clasificación como para regresión. Pero generalmente se usan en problemas de clasificación. En la década de 1960, se introdujeron por primera vez las SVM, pero luego se refinaron en 1990. Las SVM tienen su forma única de implementación en comparación con otros algoritmos de aprendizaje automático. Últimamente, son extremadamente populares debido a su capacidad para manejar múltiples variables continuas y categóricas.
---------------------
Funcionamiento de SVM
Un modelo SVM es básicamente una representación de diferentes clases en un hiperplano en un espacio multidimensional. SVM generará el hiperplano de manera iterativa para minimizar el error. El objetivo de SVM es dividir los conjuntos de datos en clases para encontrar un hiperplano marginal máximo (MMH).
---------------------
Los siguientes son conceptos importantes en SVM:
Support Vectors- Los puntos de datos que están más cerca del hiperplano se denominan vectores de soporte. La línea de separación se definirá con la ayuda de estos puntos de datos.
Hyperplane - Como podemos ver en el diagrama anterior, es un plano o espacio de decisión que se divide entre un conjunto de objetos de diferentes clases.
Margin- Puede definirse como el espacio entre dos líneas en los puntos de datos del armario de diferentes clases. Se puede calcular como la distancia perpendicular desde la línea a los vectores de apoyo. Un margen grande se considera un buen margen y un margen pequeño se considera un margen incorrecto.
---------------------
El objetivo principal de SVM es dividir los conjuntos de datos en clases para encontrar un hiperplano marginal máximo (MMH) y se puede hacer en los siguientes dos pasos:
Primero, SVM generará hiperplanos iterativamente que segrega las clases de la mejor manera.
Luego, elegirá el hiperplano que separa las clases correctamente.
---------------------
Implementando SVM en Python
Para implementar SVM en Python, comenzaremos con la importación de bibliotecas estándar de la siguiente manera:
---------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
---------------------
A continuación, estamos creando un conjunto de datos de muestra, con datos separables linealmente, de sklearn.dataset.sample_generator para la clasificación usando SVM -
---------------------
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.50)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer');
---------------------
El siguiente sería el resultado después de generar un conjunto de datos de muestra con 100 muestras y 2 grupos:

---------------------
Sabemos que SVM admite la clasificación discriminativa. divide las clases entre sí simplemente encontrando una línea en el caso de dos dimensiones o una variedad en el caso de múltiples dimensiones. Se implementa en el conjunto de datos anterior de la siguiente manera:
---------------------
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
plt.plot([0.6], [2.1], 'x', color='black', markeredgewidth=4, markersize=12)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
   plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5);
---------------------
La salida es la siguiente:

---------------------
Podemos ver en la salida anterior que hay tres separadores diferentes que discriminan perfectamente las muestras anteriores.
Como se discutió, el objetivo principal de SVM es dividir los conjuntos de datos en clases para encontrar un hiperplano marginal máximo (MMH), por lo tanto, en lugar de dibujar una línea cero entre clases, podemos dibujar alrededor de cada línea un margen de algún ancho hasta el punto más cercano. Se puede hacer de la siguiente manera:
---------------------
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
   for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
   yfit = m * xfit + b
   plt.plot(xfit, yfit, '-k')
   plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
         color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5);
---------------------

De la imagen de arriba en la salida, podemos observar fácilmente los “márgenes” dentro de los clasificadores discriminativos. SVM elegirá la línea que maximice el margen.
"""
from sklearn.datasets._samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set()
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.50)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
plt.show()
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
plt.plot([0.6], [2.1], 'x', color='black', markeredgewidth=4, markersize=12)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5)
plt.show()
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.show()
