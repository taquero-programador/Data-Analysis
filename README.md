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
    by_letter.setfedault(letter, []).append(word)

# usando setdefaultdic de collections
from collections import setdefaultdict

by_letter = setdefaultdict(list)

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

flattened = [x for tup in some_tuples for x for tup]
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
```
