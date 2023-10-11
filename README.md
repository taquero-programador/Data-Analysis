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
