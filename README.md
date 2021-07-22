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


# TODO:
* Agregar un input_to_param mapper opcional para poder mapear correctalemte la salida de un modelo con la entrada de otro modelo
* Agregar modelos multiclass y multioutput al ModuleEvaluator
* Guardar y recargar outputs desde el disco y retomar corrida desde donde dejó
* Parallelizar grafo de corridaas con subprocessos y dependencias