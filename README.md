# Data-Analysis

## Instalación
- Instalar usando Conda o manualmente
- Crear un entorno virtual.
- Instalar los paquetes esenciales: pandas, jupyter matplotlib (actualizar los paquetes).

## Conceptos básicos
### IPython
Usar `?` después de de una variable mostrara información general sobre el objeto.
A esto se le conoce como _introspección de objetos_, también puede ser usado en funciones y métodos de instancia.
```py
def add_numbers(a, b):
    """
    Add two numbers together

    Returns
    -------
    the sum: type of arguments
    """
    return a + b
```
Usar `?` con la función:
`add_numbers?`

Utilizar con los módulos para obtener una lista de las funciones disponibles:
```py
import numpy as np

np.*load*?
```

### Python
`isinstance()` permite pasar una tupla:
```py
a = 5
b = 5.4

isinstance(a, (int, float)) # True
isinstance(b, (int, float)) # True
```

Verificar si un objeto es iterable: podría usar el método mágico `__iter__` o en su lugar la
función `inter()`.
```py
def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError: # not iterable
        return False
```

Importación
```py
# some_module.py
PI = 3.14159

def f(x):
    return x + 


def g(a, b):
    return a + b
```
Acceder a las variables desde otro módulo:
```py
import some_module

result = some_module.f(5)
pi = some_module.PI

# or

from some_module import g, PI

result = g(5, PI)
```
`None` también es un valor predeterminado común para argumentos de función:
```py
def add_and_maybe_multiplay(a, b, c=None):
    result = a + b

    if c is not None:
        result = result * c
    return result
```
Fechas y horarios: `datetime` combina la información almacenada en `date` y `time`.
```py
from datetime importa datetime

dt = datetime(2011, 10, 29, 20, 30, 21)
dt.day # retorna el día
dt.minute # retorna el minuto

# extrae la fecha
dt.date()

# extrae el tiempo
dt.time()

# usar strftime()
dr.strftime("%Y-%m-%d %H:%m")

# convertir una cadena a objetos `strpime`
datetime.strpime("20091093", "%Y%m%d") # retorna un objeto strpime

# es útil reemplazar los campos de serie minutos y segundos a ceros
dt_hour = dt.replace(minute=0, second=0)

# datetime.datetime produce objetos inmutables
dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt - dt
delta
type(delta)
dt + delta

sequence = [1, 2, None, 4, None, 5]
total = 0

for value in sequence:
    if value is None:
        continue
    total += value
```

## Chapter 3: Data Structures
El operador `_` se utiliza para variables no deseadas
```py
# es preferible usar extende al crear una nueva lista (también si es muy grande). concatenación por adición
everything = []
for chunk in list_of_lists:
    everything.extend(chunk)

# es más raápido de que la alternativa concateniativa
for chunk in list_of_lists:
    everything = everything + chunk

# clasificar una lista de palabras por su primera letra
words = ["apple", "bat", "bar", "atom", "book"]
by_letter = {}

for word in words:
    letter = words[0]
    if letter not in by_letter:
        by_letter[letter] = [word] # crea una clave y una lista con una palabra
    else:
        by_letter[letter].append(word) # hace un appende de las demás palabras

# esto se puede simplificar usando `setfedault`
by_letter = {}

for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)

# usando defaultdic de collections
from collections import defaultdict

by_letter = defaultdict(list)

for word in words:
    by_letter[word[0]].append(word)

# calcular el largo único de las palabras
{len(x) for x in strings}

# eso mismo se puede hacer con map
set(map(len, strings))

# usar map junto a una función
def mult(num):
    return num * 2

list(map(mult, [1,2,3,4]))

# crear un diccionario con la ubicación de las palabras en la lista
loc_mapping = {value: index for index, value in enumerate(strings)}

# palabras con más de una "a"
all_data = [['john', 'michael', 'mary', 'steven', 'emily'],
    ['maria', 'juan', 'javier', 'natalia', 'pilar']]
names_of_interest = []

for names in all_data:
    enough_as = [name for name in names if name.count("a") >= 2]
    names_of_interest.extend(enough_as)

# hacer lo mismo con una sola compresión de lista anidada
result = [name for names in all_data for name in names if name.count("a") >= 2]

# tuplas anidadas
some_tuples = [(1,2,3), (4,5,6), (7,8,9)]

flattened = [x for tup in some_tuples for x in tup]
# una lista de compresión anidada sería muy similar a un bucle
flattened = []
for tup in some_tuples:
    for x in tup:
        flattened.append(x)

# crear un función de limpieza con re
import re

states = ["    Alabama ", "Georgia!", "Georgia", "FlOrIdA", "south carolina##", "West virginia?"]

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub("[!#?]", "", value)
        value = value.title()
        result.append(value)
    return result

# enfoque alternativo
def remove_punct(value):
    return re.sub("[!#?]", "", value)

clean_ops = [str.strip, remove_punct, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings: # recorre los elementos en la lista
        for func in ops: # recorre cada opción de la lista clean_ops
            value = func(value) # ejecuta cada acción de la lista clean_ops
        result.append(value)
    return result

clean_strings(states, clean_ops)

# usando map
for x in map(remove_punct, states):
    print(x)

# lambda permite escribir funciones que consisten en una sola instrucción
def short_function(x):
    return x * 2

equiv_anon = lambda x: x * 2

"""
lambda: a menudo es menos tipificador (y más claro) pasar una función lambda en lugar de
escribir una declaración completa o incluso asignar la función lambda a una variable local
"""
def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

inst = [4, 0, 1, 5, 6]

apply_to_list(inst, lambda x: x * 2)

# ordenar una colección de cadenas por el número de letras distintas en cada cadena
strings.sort(key=lambda x: len(set(x)))

# itertools
"""
groupby toma cualquier secuencia y una función, agrupando elementos consecutivos en
la secuencia por valor de retorno de la función
"""
import itertools

def first_letter(x):
    return x[0]

names = ["Alan", "Adam", "Wes", "Will", "Albert", "Steven"]

for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names))

# manejo de de excepciones
# manejar una excepción con float
def attemp_float(x):
    try:
        return float(x)
    except:
        return x

# manejar el ValueError
def attemp_float(x):
    try:
        return float(x)
    except ValueError:
        return x

# detectar múltiples tipos de excepción usando una tupla
def attemp_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x

"""
es posible no suprimir la excepción, pero ejecutar código independientemente
de si el código en try tiene éxito o no
"""
f = open(path, mode="w")

try:
    write_to_file(f)
finally:
    f.close()
# el objetivo de este código es siempre cerrar

# ejecutar un código solo si try tiene éxito
f = open(path, mode="w")

try:
    write_to_file(f)
except:
    print("Failed")
else:
    print("Suceeded")
finally:
    f.close()

# Archivos y el sistema operativo
path = r"examples/segismundo.txt"
f = open(path, encoding="utf-8-sig")

# de forma predeterminada, el archivo se abre en modo lectura
for line in f:
    print(line)

# el archivo sale con los marcadores de fin de línea (EOF)
lines = [x.rstrip() for x in open(path, encoding="utf-8-sig")]
lines # lo imprime sin espacios

with open(path, encoding="utf-8-sig") as f:
    lines = [x.rstrip() for x in f]

# escribir un archivo sin líneas en blanco
with open("tmp.txt", mode="w") as handle:
    handle.writelines(x for x in open(path) if len(x) > 1)

with open("tmp.txt") as f:
    lines = f.readlines()

# write escribe línea por línea y no añade el salto, mientras que writelines escribe una lista de cadenas
```

## Chapter 4: básico de NumPy
```py
# diferencia entre array y lista
import numpy as np

my_arr = np.arange(1_000_000)
my_list = list(range(1_000_000))

# comparar el tiempo
%time my_arr2 = my_arr * 2
%time my_list2 = my_list * 2

# indexación Booleana
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
data = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2], [-12, -4], [3, 4]])

names == "Bob"
# indexación
data[names == "Bob"]

# todo menos Bob
names != "Bob"

# ~ invierte una matriz booleana referenciada
~(names == "Bob")
data[~(names == "Bob")]

# ~ puede ser usado para una matriz referenciada por una variable
cond = names == "Bob"
data[~cond]

# operadores and y or
mask = (names == "Bob") | (names == "Will")
mask
data[mask]

# names[(names == "Bob") | (names == "Will")] # retorna el valor real y no la posición

# sustituye el valor o valores en el lado derecho
data[data < 0] = 0 # valores menores a cero ceran cambiados a cero
data[names != "Joe"] = 7

# indexación fancy
arr = np.zeros((8, 4))

for i in range(8):
    arr[i] = i

# seleccionar un subconjunto de filas en un orden en particular
arr[[4, 3, 0, 6]]

# funciona diferente un arrays de dimensiones
arr = np.arange(32).reshape((8, 4))
arr[[1, 5, 7, 2], [0, 3, 1, 2]] # dimensión, posición

# transposición y ejes de intercambio
arr = np.arange(15).reshape((3, 5))
arr.T # or .transpose()

# calcular el producto con numpy.dot
arr = np.array([[1, 0, 1], [1, 2, -2], [6, 3, 2], [-1, 0, 1], [1,0,1]])
np.dot(arr.T, arr)

# con @
arr.T @ arr

# generación de números pseudorandom
samples = np.random.standard_normal(size=(4, 4))

# es más rápido que random de Python
from random import normalvariate

N = 1_000_000

%time samples = [normalvariate for _ in range(N)]
%time np.random.standard_normal(N)

# permite ser configurado con seed
rng = np.random.default_rng(seed=12345)
data = rng.standard_normal((2, 3))
type(rng)

# programación orientada a matrices con matrices
# evaluar la función sqrt(x^2 + y^2)
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)

# evaluar la función
z = np.sqrt(xs ** 2 + ys ** 2)

# un ejemplo simple con matplotlib
import matplotlib.pyplot as plt

plt.imshow(z, cmap=plt.cm.gray, extent=[-5, 5, -5, 5])
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2} for a grid of values")
Text(0.5, 1.0, "Image plot of $\\sqrt{x^2 + y^2} for a grid of values")

# cerrar todas la ventanas en IPython
plt.close("all")

# np.where
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, False, False])

# tomar un valor de xarr si está en cond o tomar yarr. primero con una compresión de lista
result = [(x if x else y)
    for x, y, c in zip(xarr, yarr, cond)]

# con where
result = np.where(cond, xarr, yarr)
"""
el segundo y tercer argumento en numpy.where no es necesario que sean matrices; 
una o ambas pueden ser escalares.

una acción muy común en análisis de datos es producir una nueva matriz de valores
basados en otra matriz.
"""
arr = rng.standard_normal((4, 4))
arr > 0
np.where(arr > 0, 2, -2)

# puede combinar escalares o matrices
np.where(arr > 0, 2, arr)

"""
métodos matemáticos y estadísticos
genera algunos datos aleatorios normalmente distribuidos y calcula algunas estadísticas
agregadas.
"""
arr = rng.standard_normal((5, 4))
arr.mean()
# or
np.maen(arr)
arr.sum()

arr.mean(axis=1) # 1 columnas
arr.sum(axis=0) # 0 filas

# métodos para matrices booleanas
arr = rng.standard_normal(100)
(arr > 0).sum()
(arr <= 0).sum()

# lógica única
names = np.array(["Bob", "Will", "Bob", "Will", "Joe", "Joe"])
np.unique(names) # retorna únicos
inst = np.array([3, 3, 3, 2])
np.unique(inst)

# entrada y salida de archivos con arrays. numpy.save y numpy.load sirven para
# cargary gurdar archivos
arr = np.arange(10)
np.save("some_arrays", arr) # el archivo se guarda como .npy
np.load("some_arrays")

# guardar varias matrices en un archivo sin comprimir
np.savez("some_arr", a=arr, b=arr) # se guarda como .npz
arch = np.load("some_arr")

# se accede a cada array como un diccionario
arch["a"]

# álgebra línea
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)

# es equivalente a
np.dot(x, y)
x @ np.ones(3)

# numpy.linalg tiene un conjunto estándar de descomposiciones matriciales y cosas como inverso
# y determinante
from numpy.linalg import ibv, qr

X = rng.standard_normal((5, 5))
mat = X.T @ X
inv(mat)
mat @ inv(mat)

# ejemplo: pasos aleatorios
import random

position = 0
walk = [position]
nsteps = 1000
for _ in range(nsteps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.plot(walk[:100])

# con numpy.random
nsteps = 1000
rng = np.random.default_rng(seed=12345)
draws = rng.integers(0, 2, size=nsteps)
steps = np.where(draws == 0, 1, -1)
walk = steps.cumsum()
walk.min()
walk.max()
(np.abs(walk) >= 10).argmax()

# calcular 5000
nwalks = 5000
nsteps = 1000
draws = rng.integers(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(axis=1)
walks
walks.max()
walks.min()

# calcular el tiempo mínimo de cruce
hits30 = (np.abs(walks) >= 30).any(axis=1)
hits30.sum()

# usar una matriz booleana para seleccionar filas de walks que realmente cruzan el nivel
# absoluto 30
crossing_time = (np.abs(walks[hits30]) >= 30)
crossing_time

# calcular el tiempo mínimo de cruce promedio
crossing_time.mean()

# other example
draws = 025 * rng.standard_normal((nwalks, nsteps))
```

## Chapter 5: comenzando con Pandas
```py
import numpy as np
import pandas as pd
"""
serie: una serie es un objeto similar a una matriz unidimensional que contiene una
secuencia de valores. la serie más simple se forma a partir de solo una matriz de datos
"""
obj = pd.Series([4, 7, -5, 8])

# se puede obtener la representación de array e index
obj.array
obj.index

# crear una serie con etiquetas
obj2 = pd.Series([4, 7, -5, 8], index=["d", "b", "a", "c"])
obj2["a"]
obj2["d"] = 6
obj2[obj2 > 0]
obj2 * 2

import numpy as np

np.exp(obj2)
# otra forma de pensar acerca de una serie es como un diccionario ordenado de
# longitud fija
"b" in obj2 # True

# crear una serie a partir de un diccionario
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah"; 5000}
obj3 = pd.Series(sdata)

# una serie se puede volver a convertir a un diccionario
obj3.to_dict()

# al pasar un diccionario respeta el orden, se puede anular pasando un index en el orden deseado
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)

# California se agrega, pero el valor es NaN
pd.isna(obj4) # retorna True en los valores NaN. también obj.isna()
pd.notna(obj4) # retorna solo los False

# una característica útil es que se alinea automáticamente por etiqueta de índice en operaciones aritméticas
obj4 + obj3

# tanto el objeto Serie como su índice tinen un atributo name
obj4.name = "population"
obj4.index.name = "state"

# el índice de una serie se puede modificar mediante ua asignación
obj.index = ["Bob", "Steve", "Jeff", "Ryan"]

"""
marco de datos (DataFrame)
un DataFrame representa una tabla rectangular de datos y contiene una colección
ordenada y con nombre de columnas, cada una de las cuales puede ser un tipo de valor
diferente. El DataFrame tiene un índice de fila y columna; se puede considerar
como un diccionario de series que comparte el mismo índice.
"""
# hay muchas maneras de crear un DataFrame, aunque una de las más comunes es usar un diccionario
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
    "year": [200, 2001, 2002, 2001, 2002, 2003],
    "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)

# el método head selecciona las primeras cinco filas
frame.head() # or frame.head(int)

frame.tail() # retorna las últimas cinco filas

# pasar una columna que no está contenida en el diccionario, aparecera con valores faltantes
frame2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])

# se puede recuperar una columna por notación de diccionario o notación de punto
frame2["year"]
frame2.year

# se puede usar `iloc` y `loc` para recuperar las columnas
frame2.loc[1]
frame2.iloc[2]

# las columnas se puede modificar pos asignación
frame2.debt = 16.5 # or frame2["debt"] = 16.5
frame2.debt = np.arange(6.)

# pasar una serie se añade según su index
val = pd.Series([-1.2, -1.5, -1.7], index=[2, 4, 5])
frame2["debt"] = val

# usar del para eliminar una columna. no se puede usar la notación de punto para alterar una columna
frame2["eastern"] = frame2["state"] == "Ohio" # crea una columna con valores booleanos
del frame2["eastern"]

# otra forma común de datos es un diccionario anidado de diccionarios
populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
    "Nevada": {2001: 2.4, 2002: 2.9}}

# si el diccionario anidado se pasa al DataFrame, Pandas interpretara la claves externas como columnas
# y las claves internas como índices de fila
frame3 = pd.DataFrame(populations)

# para transponer el DataFrame
frame3.T

# las claves internas de un diccionario se combinan para formar el índice de resultado
# esto no es cierto si se especifica un índice explcícito
pd.DataFrame(populations, index=[2001, 2002, 2003])

# los diccionarios de serie de tratan de la misma manera
pdata = {"Ohio": frame3["Ohio"][:-1],
    "Nevada": frame3["Nevada"][:2]}
pd.DataFrame(pdata)

# check
frame3.index.name = "year"
frame3.columns.name = "state"

# a diferencia de las series,DataFrame no tiene un atributo name

"""
si las columnas del DataFrame son diferentes tipos de datos, se elegirá el tipo
de datos de la matriz devuelta para acomodar todas las columnas
"""
frame.to_numpy()
```
