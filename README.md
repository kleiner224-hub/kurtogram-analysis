# Kurtogram-Analysis

Implementación en Python del algoritmo **Fast Kurtogram** de Jérôme Antoni
(port del código MATLAB original de MathWorks File Exchange, realizado por
Daniel Newman en 2019), orientado al diagnóstico de fallas en maquinaria
rotativa mediante análisis de vibraciones.

El kurtograma localiza la banda de frecuencia donde la señal es más impulsiva
(mayor curtosis espectral), que es donde suelen manifestarse los impactos de
una falla en rodamientos. Esa banda se usa luego para demodular la señal y
obtener el espectro de envolvente.

## Contenido

- `kurtogram.py`: función `fast_kurtogram(x, fs, nlevel)`. Devuelve la matriz
  del kurtograma, los niveles, las frecuencias y la señal filtrada en la banda
  óptima.
- `freq_tools.py`: utilidades de procesamiento de señal (FFT promediada tipo
  Welch, demodulación de envolvente, curtosis espectral por STFT, deconvolución
  MED, detección de picos y de velocidad real de giro).
- `Kurtogram-Example.ipynb`: notebook de ejemplo con el flujo completo.
- `aoyu_example_data.txt`: señal de vibración exportada desde LabVIEW
  (100 kHz, máquina a 1000 RPM).

## Uso rápido

```python
import numpy as np
from kurtogram import fast_kurtogram

data = np.genfromtxt('aoyu_example_data.txt', skip_header=23, delimiter='\n')
Kwav, Level_w, freq_w, c = fast_kurtogram(data, fs=100e3, nlevel=6)
```

Dependencias: `numpy`, `scipy`, `matplotlib`.

## Licencia

El código de `kurtogram.py` conserva la licencia BSD del autor original
(Jérôme Antoni, 2015). Ver el encabezado del archivo.
