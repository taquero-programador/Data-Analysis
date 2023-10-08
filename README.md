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
