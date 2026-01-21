# Práctica: Modelo Seq2Seq con Atención de Bahdanau (LSTM)

## Descripción general

En esta práctica se implementa y evalúa un modelo *sequence-to-sequence* (Seq2Seq) basado en redes **LSTM** con **mecanismo de atención de Bahdanau**, aplicado a una tarea de traducción simplificada entre secuencias.

El objetivo principal no es construir un sistema de traducción realista, sino **comprender y demostrar experimentalmente cómo el mecanismo de atención mejora el rendimiento de los modelos Seq2Seq**, permitiendo al decodificador centrarse dinámicamente en distintas partes de la secuencia de entrada en cada paso de generación.

Este enfoque supera la limitación clásica de los modelos Seq2Seq básicos, que utilizan un único vector de contexto fijo para representar toda la secuencia de entrada.

---

## Objetivos de la práctica

- Implementar un modelo **Encoder–Decoder** basado en LSTM.
- Incorporar el **mecanismo de atención de Bahdanau**.
- Entrenar el modelo sobre un conjunto de datos sintético.
- Analizar el comportamiento del modelo durante el entrenamiento.
- Evaluar cualitativamente los resultados obtenidos.

---


---

## Descripción de los componentes principales

### Dataset y vocabulario

El archivo `dataset.py` define:

- La clase `TranslationDataset`, encargada de cargar y preparar los pares de secuencias.
- La clase `Vocab`, que construye el vocabulario y gestiona la conversión entre tokens y índices.
- La función `collate_fn`, utilizada por el `DataLoader` para crear lotes con padding.

---

### Generación de datos

En `data/dataGenerator.py` se generan secuencias sintéticas de entrada y salida, lo que permite:

- Controlar la dificultad del problema.
- Analizar el comportamiento del modelo sin depender de datos reales complejos.
- Centrarse en el estudio del mecanismo de atención.

---

### Modelo Seq2Seq con atención

El archivo `model.py` contiene:

- **Encoder**: LSTM que procesa la secuencia de entrada.
- **Decoder**: LSTM que genera la secuencia de salida.
- **Mecanismo de atención de Bahdanau**, que calcula un contexto dinámico en cada paso del decodificador.
- **Modelo Seq2Seq**, que integra todos los componentes anteriores.

La atención permite ponderar los estados ocultos del encoder en función del estado actual del decoder, mejorando la alineación entre entrada y salida.

---

### Entrenamiento

El archivo `main.py` gestiona:

- La carga del dataset.
- La inicialización del modelo.
- El proceso de entrenamiento.
- El cálculo de la función de pérdida.
- El seguimiento de la evolución del error por época.

Durante el entrenamiento se observa una disminución progresiva de la pérdida, indicando que el modelo aprende correctamente la tarea propuesta.

---

## Ejecución del proyecto

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Resultados

El entrenamiento muestra una reducción consistente de la función de pérdida, lo que indica una correcta convergencia del modelo.

El uso del mecanismo de atención permite al decodificador centrarse en distintas posiciones de la secuencia de entrada, produciendo salidas más coherentes que un modelo Seq2Seq sin atención.

---

## Conclusiones

Esta práctica demuestra de forma experimental que el mecanismo de atención de Bahdanau mejora significativamente el funcionamiento de los modelos Seq2Seq, al eliminar la necesidad de condensar toda la información de la secuencia de entrada en un único vector.

El enfoque resulta especialmente relevante en tareas donde la longitud de las secuencias es elevada o donde la alineación entre entrada y salida es clave.


