# ML_ConPipe
A configuration based work pipeline for ML experiments

# Install ML_ConPipe package 
```
pip install mlconpipe
```

# Simple example

experiment.yaml
```
Aca va todos los parametros del config
```

feature_extraction.py
```
Ejemplo de un custom feature extraction
```

custom_model.py
```
Ejemplo de un modelo custom
```

# Configuraciones:

### force_not_rerun
si es true, entonces solo ejecuta ese nodo una sola vez y luego ya no lo vuelve a ejecutar. Hay que tener cuidado si se borran los output igual no se vuelve a correr a menos que se elimine todo el estado del pipeline o el de ese nodo. Util para nodos que tardan siglos en correr y que solo se quiere correr una sola vez.

### bypass
Esto skipea la ejecusión del nodo asumiendo que ya fue ejecutado, ojo que si el output no existe puede estallar. Este paremetro es parecido a force_not_rerun pero cambia en el hecho que directamente lo bypasea y no lo corre ni una vez, sirve en casos especiales donde un nodo se desea que este por completitud pero su output ya fue calculado o algo así.

**Note:** Si bypass es True, entonces force_not_rerun no tienen ningún efecto especial ya que este parámetro es más fuerte

### dependencies
Marca dependencias que puede tener con otros nodos aunque no consuma output de esos nodos, esto es util para cuando no se utiliza el pipeline como mecanismo de traspaso de datos entre los nodos (por ejemplo porque se usa una base de datos donde los nodos van leyendo y escribiendo), en estos casos el pipeline no ejecutaría a los nodos en el orden correcto porque no se conectan sus outputs y estallaría.