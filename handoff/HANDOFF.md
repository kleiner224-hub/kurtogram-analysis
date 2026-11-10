# Traspaso: Fast Kurtogram + envolvente (para Vibralert / PicoDiag)

Paquete autocontenido para reutilizar en otra sesión o proyecto. No depende
del resto del repositorio salvo para los datos de ejemplo.

## Archivos

| Archivo | Qué es |
|---|---|
| `kurtogram_envelope.py` | Módulo único: `fast_kurtogram` (port fiel de Antoni, corregido) + pipeline de envolvente + CLI. |
| `config.json` | Parámetros del pipeline. La geometría del rodamiento es de EJEMPLO (tipo 6205): sustituir por la real. |
| `results.json` | Salida de la última ejecución sobre `aoyu_example_data.txt`. |
| `out_kurtogram.png`, `out_envelope.png` | Figuras de esa ejecución. |

## Pipeline (lo que hace `run_pipeline`)

1. Señal cruda `x` a `fs` Hz, se le quita la media.
2. `fast_kurtogram(x, fs, nlevel)`: banco de filtros binario/ternario, curtosis
   por banda y nivel. Devuelve frecuencia central `fc_hz`, ancho `bandwidth_hz`
   y curtosis máxima.
3. Filtro paso banda Butterworth en `[fc - bw/2, fc + bw/2]` (filtfilt, fase cero).
4. Envolvente por transformada de Hilbert, diezmada a `env_fs` (2,5 kHz por defecto).
5. Espectro de envolvente por Welch, resolución `env_fs / nperseg` (0,3 Hz por defecto).
6. Frecuencias de falla a partir de la geometría: FTF, BPFO, BPFI, BSF, más 1X.
7. Para cada una, busca pico en los primeros `harmonics` armónicos con tolerancia
   `tolerance_pct`, calcula SNR = pico / mediana local, y marca detección si
   SNR >= `min_snr` **y el pico no coincide con un armónico de 1X**.

## Uso

```bash
pip install numpy scipy matplotlib
python kurtogram_envelope.py config.json --plot
```

```python
from kurtogram_envelope import run_pipeline
res = run_pipeline(x, fs=100e3, nlevel=6, speed_hz=1000/60,
                   bearing=dict(n_balls=9, ball_d=7.94, pitch_d=39.04, contact_deg=0))
res['kurtogram']        # banda elegida
res['matches']['BPFO']  # picos encontrados
res['fault_detected']
```

## Resultado sobre los datos de ejemplo

- Banda óptima: fc 44 792 Hz, ancho 2 083 Hz, curtosis 90,6 (nivel 4,58).
- 1X a 16,7 Hz con armónicos muy claros (SNR 29).
- Líneas dominantes en el espectro de envolvente a ~125 Hz y ~250 Hz que
  la geometría de ejemplo NO explica. Con la geometría real del rodamiento
  habría que comprobar si corresponden a BPFO o BPFI.
- Los "detectados" de BPFO/BPFI/BSF en `results.json` no son concluyentes:
  la geometría es inventada y varios picos caen sobre armónicos de 1X.

## Reglas de fiabilidad aprendidas

- Resolución del espectro de envolvente: diezmar la envolvente antes de Welch.
  Sin eso, con fs 100 kHz y nperseg 4096 la resolución era 24 Hz y no se veían
  ni 1X ni FTF.
- Excluir picos que coinciden con k·1X: en esta señal el peine de 1X genera
  falsos positivos en BPFI (183 Hz = 11·1X, 267 Hz = 16·1X) y BSF.
- La curtosis se dispara con impulsos aislados (golpes, ruido eléctrico).
  Validar siempre que el espectro de envolvente tenga picos periódicos.
- Para fallas incipientes o mucho ruido, pre-blanquear (MED) antes del kurtograma.
- Coste: ~segundos por 1,9 M muestras en Python puro. Para tiempo real,
  calcular la banda periódicamente y fijarla entre recálculos.

## Correcciones respecto al repo original

- `_kurt` con 'kurt1' usaba una variable inexistente.
- `fast_kurtogram` modificaba la señal de entrada en sitio.
- La frecuencia central impresa por el original mezclaba Hz con frecuencia
  normalizada (daba 1042 Hz); ahora sigue la fórmula de MATLAB (44 792 Hz).
- El notebook llamaba `signal.welch` con el módulo importado como `sig`.
- Las funciones `kurtogram` y `spectralKurtosis` de `freq_tools.py` son una
  aproximación con fórmula modificada reconocida por el autor: no usarlas.

## Referencias

- J. Antoni, "Fast computation of the kurtogram for the detection of
  transient faults", MSSP 2007. Código MATLAB: File Exchange 48912.
- R. B. Randall, "Vibration-Based Condition Monitoring", cap. 5 (envolvente).
- Alternativas más robustas a impulsos aleatorios: Infogram (Antoni 2016),
  Autogram (Moshrefzadeh & Fasana 2018).
