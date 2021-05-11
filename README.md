# unsupervised-solver

## Requerimientos
Para correr el solver, es necesario tener instalado `Python 3`.

Además, deben instalarse `numpy`, `matplotlib`, `sklearn` y `pandas`, lo cual se puede lograr con

- `python3 -m pip install numpy`
- `python3 -m pip install matplotlib`
- `python3 -m pip install sklearn`
- `python3 -m pip install pandas`

### Versiones
Para el desarrollo, se utilizó la versión `Python 3.8.5`.

## Ejecución PCA
Para ejecutar el análisis PCA sobre `data_file`, se debe ejecutar desde la raíz del proyecto

`python3 pca.py`

## Ejecución Oja
Para ejecutar el análisis con la regla de Oja sobre `data_file`, se debe ejecutar desde la raíz del proyecto

`python3 oja.py`

## Ejecución Kohonen
Para ejecutar el análisis con la red de Kohonen sobre `data_file`, se debe ejecutar desde la raíz del proyecto

`python3 kohonen.py`

## Ejecución Hopfield
Para ejecutar el análisis con la red de Hopfield sobre los patrones almacenados en `pattern_dir`, se debe ejecutar desde la raíz del proyecto

`python3 hopfield.py`

# Configuraciones
Todas las configuraciones de ejecución se establecen en el archivo `config.json`. A continuación, se explica a qué hace referencia cada configuración:
- **data_file** indica la dirección del archivo `.csv` sobre el cual se desea realizar el análisis de componentes principales (Eg: europe.csv).
- **oja** contiene las configuraciones de la regla de Oja:
    - **eta** indica el valor de eta a utilizar.
    - **epoch** indica la cantidad de épocas a analizar.
- **kohonen** contiene las configuraciones de la red de Kohonen:
    - **k** indica el lado de la matriz de salida (la matriz será de k x k).
    - **c** indica la rapidez de decrecimiento del radio, que estará como factor en un exponente negativo de e. A mayor valor de c, más rápido disminuirá el radio.
    - **w0_random_init** indica si se utilizan pesos iniciales aleatorios entre 0 y 1 (false) o si se toman muestras al azar para los mismos (true).
    - **kxk_iterations** indica el factor para calcular el número de épocas. Se realizarán kxk_iterations * k * k épocas.
    - **noramlize_data** indica si, luego de estandarizar los datos de entrada, se los normaliza (true) o no (false).    
- **hopfield** contiene las configuraciones de la red de Hopfield:
    - **pattern_dir** indica el directorio donde se encuentran los archivos con los patrones de 5x5 conocidos.
    - **mutation_prob** indica la probabilidad de mutar para cada píxel del patrón seleccionado al azar.
    - **max_iterations** indica el número máximo de iteraciones (para evitar loops infinitos).
- **plot** indica si se desean (true) o no (false) realizar ploteos.


### Ejemplo 1
```json
{
	"data_file": "europe.csv",

	"oja": {
		"eta": 0.02,
		"epoch": 500
	},

	"kohonen": {
		"k": 5,
		"c": 0.001,
		"w0_random_init": false,
		"kxk_iterations": 50,
		"normalize_data": true
	},

	"hopfield": {
		"pattern_dir": "known_patterns",
		"mutation_prob": 0.1,
		"max_iterations": 20
	},

	"plot": true
}

```

## Presentación
Link a la presentación completa: 