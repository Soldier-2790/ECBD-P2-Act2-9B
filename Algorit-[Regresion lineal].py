"""
Algoritmos de regresión - Regresión lineal
Introducción a la regresión lineal
La regresión lineal puede definirse como el modelo estadístico que analiza la relación lineal entre una variable dependiente con un conjunto dado de variables independientes. La relación lineal entre variables significa que cuando el valor de una o más variables independientes cambia (aumenta o disminuye), el valor de la variable dependiente también cambia en consecuencia (aumenta o disminuye).
Matemáticamente, la relación se puede representar con la ayuda de la siguiente ecuación:
Y = mX + b
Aquí, Y es la variable dependiente que estamos tratando de predecir.
X es la variable dependiente que usamos para hacer predicciones.
m es la pendiente de la línea de regresión que representa el efecto que X tiene sobre Y
b es una constante, conocida como intersección con el eje Y. Si X = 0, Y sería igual ab.
Además, la relación lineal puede ser de naturaleza positiva o negativa, como se explica a continuación:
-----------------
Relación lineal positiva
Una relación lineal se llamará positiva si aumentan tanto la variable independiente como la dependiente. Se puede entender con la ayuda del siguiente gráfico:
-----------------
Relación lineal negativa
Una relación lineal se llamará positiva si los aumentos independientes y la variable dependiente disminuyen. Se puede entender con la ayuda del siguiente gráfico:
-----------------
Tipos de regresión lineal
La regresión lineal es de los dos tipos siguientes:
Regresión lineal simple
Regresión lineal múltiple
-----------------
Regresión lineal simple (SLR)
Es la versión más básica de regresión lineal que predice una respuesta utilizando una sola característica. El supuesto en SLR es que las dos variables están relacionadas linealmente.
Implementación de Python
Podemos implementar SLR en Python de dos maneras, una es para proporcionar su propio conjunto de datos y la otra es usar el conjunto de datos de la biblioteca de python scikit-learn.

Example 1 - En el siguiente ejemplo de implementación de Python, estamos usando nuestro propio conjunto de datos.

Primero, comenzaremos con la importación de paquetes necesarios de la siguiente manera:

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
A continuación, defina una función que calculará los valores importantes para SLR -

def coef_estimation(x, y):
La siguiente línea de secuencia de comandos dará un número de observaciones n -

n = np.size(x)
La media de los vectores xey se puede calcular de la siguiente manera:

m_x, m_y = np.mean(x), np.mean(y)
Podemos encontrar la desviación cruzada y la desviación de x de la siguiente manera:

SS_xy = np.sum(y*x) - n*m_y*m_x
SS_xx = np.sum(x*x) - n*m_x*m_x
A continuación, los coeficientes de regresión, es decir, b se pueden calcular de la siguiente manera:

b_1 = SS_xy / SS_xx
b_0 = m_y - b_1*m_x
return(b_0, b_1)
A continuación, necesitamos definir una función que trazará la línea de regresión y predecirá el vector de respuesta:

def plot_regression_line(x, y, b):
La siguiente línea de secuencia de comandos trazará los puntos reales como diagrama de dispersión:

plt.scatter(x, y, color = "m", marker = "o", s = 30)
La siguiente línea de secuencia de comandos predecirá el vector de respuesta:

y_pred = b[0] + b[1]*x
Las siguientes líneas de secuencia de comandos trazarán la línea de regresión y les colocarán las etiquetas:

plt.plot(x, y_pred, color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
Por último, necesitamos definir la función main () para proporcionar un conjunto de datos y llamar a la función que definimos anteriormente:

def main():
   x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
   y = np.array([100, 300, 350, 500, 750, 800, 850, 900, 1050, 1250])
   b = coef_estimation(x, y)
   print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))
   plot_regression_line(x, y, b)
   
if __name__ == "__main__":
main()
Salida
Estimated coefficients:
b_0 = 154.5454545454545
b_1 = 117.87878787878788
"""
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


def coef_estimation(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m", marker="o", s=30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([100, 300, 350, 500, 750, 800, 850, 900, 1050, 1250])
    b = coef_estimation(x, y)
    print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x, y, b)


if __name__ == "__main__":
    main()
