# perceptron-solver
# Perceptron Simple y Multicapa

## Requerimientos
Para correr el solver, es necesario tener instalado `Python 3`.

Además, debe instalarse `matplotlib`, lo cual se puede lograr con

`python3 -m pip install matplotlib`

### Versiones
Para el desarrollo, se utilizó la versión `Python 3.8.5`

## Ejecución
Para ejecutar el programa, se debe ejecutar desde la raíz del proyecto

`python3 main.py`

## Configuraciones
Todas las configuraciones de ejecución se establecen en el archivo `config.json`. A continuación, se explica a qué hace referencia cada configuración:
- **training_file** indica de donde se toma el conjunto de entrenamiento
- **expected_out_file** indica de dónde se toma la salida esperada del perceptrón

- **training_ratio** indica que porcentaje del training_file se usa para entrenar, quedando el restante para probar
- **cross_validation** indica el boolean de si se desea hacer validación cruzada o no
- **delta_metrics** indica el `rel_tol` (tolerancia relativa) de igualdad al evaluar el rendimiento del perceptrón mediante math.isclose()

- **eta** indica el valor exacto del eta a utilizar (grado de aprendizaje)
- **beta** indica ael multiplicador de x al utilizar una función no lineal
- **system_threshold** indica el valor constante asignado al primer elemento de los conjuntos que se multiplicarán con w, cuyo primer elemento es w0
- **error_threshold** indica el delta con el cual se considerará que el error ne el entrenamiento llega a un mínimo con el que deseamos terminar de iterar
- **count_threshold** indica el máximo de iteraciones tras el cual terminar el entrenamiento
- **epoch_training** indica el boolean de si se desea entrenar por épocas o no
- **system** indica la función de activación, cuyos valores posibles son:
    - `sign`        --> función signo
    - `linear`      --> función identidad (lineal)
    - `tanh`        --> función tangente hiperbólica (no lineal)

- **layout** es un array que indica las capas ocultas y sus dimensiones respectivas, irá vacío de no haber capas ocultas (no multicapa)

- **randomize_w** indica si se quiere randomizar la inicialización de los pesos
- **randomize_w_ref** indica de dónde a dónde se hace la randomización (limites)
- **reset_w** indica si se quiere reiniciar el vector de pesos durante el entrenamiento
- **reset_w_iterations** indica cada cuántas iteraciones se querría hacer ese reinicio, donde este valor se multiplica por la longitud del conjunto de entrenamiento dando origen al total de iteraciones tras el cual reiniciar W

- **retro_error_enhance** indica el boolean de si se desea calcular el error con la técnica logarítmica o no

- **momentum** indica el boolean de si se desea aplicar la técnica de momentum o no
- **momentum_alpha** indica el valor del coeficiente alfa que se usa para momentum

- **general_adaptive_eta** indica si se desea aplicar la técnica de eta adaptativo o no
- **a** indica el coeficiente `a` al usar eta adaptativo
- **delta_error_decrease_iterations** indica el total de iteraciones tras el cual se considera que el error esta decreciendo (para el eta adaptativo)
- **b** indica el coeficiente `b` al usar eta adaptativo
- **delta_error_increase_iterations** indica el total de iteraciones tras el cual se considera que el error esta creciendo (para el eta adaptativo)

- **print_each_cross_validation** indica el boolean de si se desea ver en terminal el resultado de cada caso en la validación cruzada
- **normalize_out** indica el boolean de si se desea normalizar la salida del perceptrón
- **trust_min** indica el margen central de discriminación al normalizar la salida del perceptrón, de ser 
- **float_rounding_dec** indica la cantidad de decimales en los redondeos

### Ejemplo 1
```json
{
	"training_file": "inputs/ej1-entrenamiento.txt",
	"expected_out_file": "inputs/ej1a-salida-deseada.txt",

	"training_ratio": 50,
	"cross_validation": false,
	"delta_metrics" : 0.05,

	"eta": 0.001,
	"beta": 0.5,
	"system_threshold": 1,
	"error_threshold": 0.0001,
	"count_threshold": 100,
	"epoch_training": false,
	"system": "sign",

	"layout": [],

	"randomize_w": true,
	"randomize_w_ref": 100.0,
	"reset_w": false,
	"reset_w_iterations": 100,

	"retro_error_enhance": false,

	"momentum": false,
	"momentum_alpha": 0.9,

	"general_adaptive_eta": false,
	"a": 0.05,
	"delta_error_decrease_iterations": 10,
	"b": 0.1,
	"delta_error_increase_iterations": 15,

	"print_each_cross_validation": true,
	"normalize_out": false,
	"trust_min": 0.15,
	"float_rounding_dec": 3,

	"plot": true
}
```

### Ejemplo 2
```json
{
	"training_file": "inputs/ej2-entrenamiento.txt",
	"expected_out_file": "inputs/ej2-salida-deseada.txt",

	"training_ratio": 50,
	"cross_validation": false,
	"delta_metrics" : 0.1,

	"eta": 0.001,
	"beta": 0.5,
	"system_threshold": 1,
	"error_threshold": 0.0001,
	"count_threshold": 10000,
	"epoch_training": false,
	"system": "tanh",

	"layout": [],

	"randomize_w": false,
	"randomize_w_ref": 100.0,
	"reset_w": false,
	"reset_w_iterations": 100,

	"retro_error_enhance": false,

	"momentum": true,
	"momentum_alpha": 0.9,

	"general_adaptive_eta": true,
	"a": 0.05,
	"delta_error_decrease_iterations": 10,
	"b": 0.1,
	"delta_error_increase_iterations": 15,

	"print_each_cross_validation": true,
	"normalize_out": false,
	"trust_min": 0.15,
	"float_rounding_dec": 3,

	"plot": true
}
```

### Ejemplo 3
```json
{
	"training_file": "inputs/ej3-entrenamiento.txt",
	"expected_out_file": "inputs/ej3-salida-deseada.txt",

	"training_ratio": 0,
	"cross_validation": false,
	"delta_metrics" : 0.1,

	"eta": 0.01,
	"beta": 0.5,
	"system_threshold": 1,
	"error_threshold": 0.0,
	"count_threshold": 100,
	"epoch_training": true,
	"system": "tanh",

	"layout": [10,10,10],

	"randomize_w": true,
	"randomize_w_ref": 1.0,
	"reset_w": false,
	"reset_w_iterations": 100,

	"retro_error_enhance": false,

	"momentum": true,
	"momentum_alpha": 0.9,

	"general_adaptive_eta": false,
	"a": 0.05,
	"delta_error_decrease_iterations": 10,
	"b": 0.1,
	"delta_error_increase_iterations": 15,

	"print_each_cross_validation": true,
	"normalize_out": true,
	"trust_min": 0.7,
	"float_rounding_dec": 3,

	"plot": true
}
```

## Presentación
Link a la presentación completa: 
https://docs.google.com/presentation/d/1p9icVkWGWdU7htXewgau19JmhgLkST0a80b1XhdO7mg/edit?usp=sharing