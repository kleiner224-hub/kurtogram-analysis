"""
kurtogram_envelope.py  --  Pipeline autocontenido: Fast Kurtogram + análisis de envolvente.

Uso:
    python kurtogram_envelope.py config.json            # procesa y escribe resultados JSON
    python kurtogram_envelope.py config.json --plot     # además guarda figuras PNG

Como librería:
    from kurtogram_envelope import run_pipeline
    res = run_pipeline(x, fs, nlevel=6, speed_hz=16.67, bearing={...})

Dependencias: numpy, scipy (matplotlib solo para --plot).

El núcleo `fast_kurtogram` es un port fiel del algoritmo de Jérôme Antoni
(MATLAB File Exchange 48912, licencia BSD, (c) 2015 Jerome Antoni; port a
Python de Daniel Newman 2019) con dos correcciones: no modifica la señal de
entrada y la variante 'kurt1' usa la variable correcta.
"""
import json
import sys
import numpy as np
from scipy.signal import firwin, lfilter, butter, filtfilt, welch, hilbert, decimate

# --------------------------------------------------------------------------- #
#  Fast Kurtogram (Antoni)                                                     #
# --------------------------------------------------------------------------- #

def fast_kurtogram(x, fs, nlevel=7, verbose=False):
    """Devuelve (Kwav, Level_w, freq_w, c, info).
    Kwav   : matriz kurtograma (2*nlevel filas x 3*2^nlevel columnas)
    Level_w: niveles de descomposición
    freq_w : eje de frecuencias en Hz
    c      : señal compleja filtrada en la banda de máxima curtosis
    info   : dict con level, fc_hz, bandwidth_hz, max_kurtosis
    """
    x = np.asarray(x, dtype=float).flatten()
    N = x.size
    if nlevel > np.log2(N) - 7:
        raise ValueError('nlevel demasiado grande para la longitud de la señal')
    x = x - np.mean(x)

    Nf = 16
    fc = 0.4
    h = firwin(Nf + 1, fc) * np.exp(2j * np.pi * np.arange(Nf + 1) * 0.125)
    n = np.arange(2, Nf + 2)
    g = h[(1 - n) % Nf] * (-1.) ** (1 - n)
    Nf = int(np.trunc(3 / 2 * Nf))
    h1 = firwin(Nf + 1, 2 / 3 * fc) * np.exp(2j * np.pi * np.arange(Nf + 1) * 0.25 / 3)
    h2 = h1 * np.exp(2j * np.pi * np.arange(Nf + 1) / 6)
    h3 = h1 * np.exp(2j * np.pi * np.arange(Nf + 1) / 3)

    Kwav = np.clip(_K_wpQ(x, h, g, h1, h2, h3, nlevel, 'kurt2'), 0, np.inf)
    Level_w = np.arange(1, nlevel + 1)
    Level_w = np.vstack((Level_w, Level_w + np.log2(3) - 1)).flatten()
    Level_w = np.sort(np.insert(Level_w, 0, 0)[:2 * nlevel])
    freq_w = fs * (np.arange(3 * 2 ** nlevel) / (3 * 2 ** (nlevel + 1)) + 1 / (3 * 2 ** (2 + nlevel)))

    row_max = Kwav[np.arange(Kwav.shape[0]), np.argmax(Kwav, axis=1)]
    li = int(np.argmax(row_max))
    max_kurt = float(row_max[li])
    level_max = Level_w[li]
    bandwidth = fs * 2 ** (-(level_max + 1))
    J = np.argmax(Kwav[li, :])
    fi = J / 3 / 2 ** (nlevel + 1) + 2 ** (-2 - level_max)
    fc_hz = fs * fi

    info = dict(level=float(level_max), fc_hz=float(fc_hz),
                bandwidth_hz=float(bandwidth), max_kurtosis=max_kurt)
    if verbose:
        print('Kurtogram -> nivel {level:.3f}, fc {fc_hz:.1f} Hz, BW {bandwidth_hz:.1f} Hz, '
              'kurtosis {max_kurtosis:.2f}'.format(**info))

    c = _find_wav_kurt(x, h, g, h1, h2, h3, level_max, fi)
    return Kwav, Level_w, freq_w, c, info


def _kurt(v, opt='kurt2'):
    eps = 2.2204e-16
    v = np.asarray(v).flatten()
    if np.all(v == 0):
        return 0.0
    v = v - np.mean(v)
    if opt == 'kurt2':
        E = np.mean(np.abs(v) ** 2)
        if E < eps:
            return 0.0
        K = np.mean(np.abs(v) ** 4) / E ** 2
        return K - 3 if np.all(np.isreal(v)) else K - 2
    E = np.mean(np.abs(v))
    if E < eps:
        return 0.0
    K = np.mean(np.abs(v) ** 2) / E ** 2
    return K - 1.57 if np.all(np.isreal(v)) else K - 1.27


def _DBFB(x, h, g):
    x = x.flatten(); N = x.size
    a = lfilter(h, 1, x)[1:N:2]
    d = lfilter(g, 1, x)[1:N:2]
    return a, d


def _TBFB(x, h1, h2, h3):
    x = x.flatten(); N = x.size
    return (lfilter(h1, 1, x)[2:N:3], lfilter(h2, 1, x)[2:N:3], lfilter(h3, 1, x)[2:N:3])


def _K_wpQ(x, h, g, h1, h2, h3, nlevel, opt):
    KD, KQ = _K_wpQ_local(x.flatten(), h, g, h1, h2, h3, nlevel, opt, nlevel)
    K = np.zeros((2 * nlevel, 3 * 2 ** nlevel))
    K[0, :] = KD[0, :]
    for i in range(1, nlevel):
        K[2 * i - 1, :] = KD[i, :]
        K[2 * i, :] = KQ[i - 1, :]
    K[2 * nlevel - 1, :] = KD[nlevel, :]
    return K


def _K_wpQ_local(x, h, g, h1, h2, h3, nlevel, opt, level):
    a, d = _DBFB(x, h, g)
    N = a.size
    d = d * (-1) ** np.arange(1, N + 1)
    Lh, Lg = h.size, g.size
    K1 = _kurt(a[Lh - 1:], opt)
    K2 = _kurt(d[Lg - 1:], opt)
    if level > 2:
        a1, a2, a3 = _TBFB(a, h1, h2, h3)
        d1, d2, d3 = _TBFB(d, h1, h2, h3)
        Ka1, Ka2, Ka3 = (_kurt(v[Lh - 1:], opt) for v in (a1, a2, a3))
        Kd1, Kd2, Kd3 = (_kurt(v[Lh - 1:], opt) for v in (d1, d2, d3))
    else:
        Ka1 = Ka2 = Ka3 = Kd1 = Kd2 = Kd3 = 0.0
    if level == 1:
        K = np.concatenate((K1 * np.ones(3), K2 * np.ones(3)))
        KQ = np.array([Ka1, Ka2, Ka3, Kd1, Kd2, Kd3])
    else:
        Ka, KaQ = _K_wpQ_local(a, h, g, h1, h2, h3, nlevel, opt, level - 1)
        Kd, KdQ = _K_wpQ_local(d, h, g, h1, h2, h3, nlevel, opt, level - 1)
        K = np.vstack((np.concatenate([K1 * np.ones(Ka.shape[-1]), K2 * np.ones(Kd.shape[-1])]),
                       np.hstack((Ka, Kd))))
        Long = int(2 / 6 * KaQ.shape[-1])
        KQ = np.vstack((np.concatenate([k * np.ones(Long) for k in (Ka1, Ka2, Ka3, Kd1, Kd2, Kd3)]),
                        np.hstack((KaQ, KdQ))))
    if level == nlevel:
        K1 = _kurt(x, opt)
        K = np.vstack((K1 * np.ones(K.shape[-1]), K))
        a1, a2, a3 = _TBFB(x, h1, h2, h3)
        Long = int(1 / 3 * KQ.shape[-1])
        top = np.concatenate([_kurt(v[Lh - 1:], opt) * np.ones(Long) for v in (a1, a2, a3)])
        KQ = np.vstack((top, KQ[:-2, :]))
    return K, KQ


def _binary(i, k):
    k = int(k)
    a = np.zeros(k); temp = i
    for l in np.arange(k)[::-1]:
        a[-(l + 1)] = np.trunc(temp / 2 ** l)
        temp -= a[-(l + 1)] * 2 ** l
    return a


def _find_wav_kurt(x, h, g, h1, h2, h3, Sc, Fr):
    level = np.trunc(Sc) + (np.remainder(Sc, 1) >= 0.5) * (np.log2(3) - 1)
    Bw = 2 ** (-level - 1)
    freq_w = np.arange(2 ** level) / (2 ** (level + 1)) + Bw / 2
    J = np.argmin(np.abs(freq_w - Fr))
    i = np.round(freq_w[J] / Bw - 1 / 2)
    if np.remainder(level, 1) == 0:
        acoeff, bcoeff, tl = _binary(i, level), None, int(level)
    else:
        i2 = np.trunc(i / 3)
        tl = int(np.trunc(level) - 1)
        acoeff, bcoeff = _binary(i2, tl), int(i - i2 * 3)
    acoeff = acoeff[::-1]
    return _wpQ_filt(x, h, g, h1, h2, h3, acoeff, bcoeff, tl)


def _wpQ_filt(x, h, g, h1, h2, h3, acoeff, bcoeff, level):
    x = x.flatten()
    if acoeff.size == 0:
        if bcoeff is None:
            return x
        return _TBFB(x, h1, h2, h3)[bcoeff][h1.size - 1:]
    a, d = _DBFB(x, h, g)
    d = d * (-1) ** np.arange(1, a.size + 1)
    if level == 1:
        src = a if acoeff[0] == 0 else d
        if bcoeff is None:
            return src[(h if acoeff[0] == 0 else g).size - 1:]
        return _TBFB(src, h1, h2, h3)[bcoeff][h1.size - 1:]
    nxt = a if acoeff[level - 1] == 0 else d
    return _wpQ_filt(nxt, h, g, h1, h2, h3, acoeff, bcoeff, level - 1)

# --------------------------------------------------------------------------- #
#  Pipeline de envolvente                                                      #
# --------------------------------------------------------------------------- #

def bearing_frequencies(speed_hz, n_balls, ball_d, pitch_d, contact_deg=0.0):
    """BPFO, BPFI, BSF, FTF en Hz a partir de la geometría del rodamiento."""
    r = ball_d / pitch_d * np.cos(np.deg2rad(contact_deg))
    return dict(
        FTF=0.5 * speed_hz * (1 - r),
        BPFO=0.5 * n_balls * speed_hz * (1 - r),
        BPFI=0.5 * n_balls * speed_hz * (1 + r),
        BSF=0.5 * pitch_d / ball_d * speed_hz * (1 - r ** 2),
    )


def envelope_spectrum(x, fs, fc_hz, bw_hz, env_fs=2500.0, nperseg=8192, order=3):
    """Filtra x en [fc-bw/2, fc+bw/2], toma la envolvente (Hilbert), la
    diezma a env_fs Hz y devuelve su espectro (Welch) y la envolvente.
    Resolucion espectral = env_fs / nperseg (0.3 Hz con los valores por defecto)."""
    nyq = fs / 2
    lo = max(fc_hz - bw_hz / 2, 1.0) / nyq
    hi = min(fc_hz + bw_hz / 2, nyq * 0.99) / nyq
    b, a = butter(order, [lo, hi], btype='band')
    xb = filtfilt(b, a, x)
    env = np.abs(hilbert(xb))
    q = int(fs // env_fs)
    if q > 1:
        while q > 13:                      # decimate recomienda factores <= 13
            env = decimate(env, 10, ftype='fir'); q //= 10
        env = decimate(env, q, ftype='fir')
        fs_env = fs / int(fs // env_fs)
    else:
        fs_env = fs
    env = env - np.mean(env)
    f, p = welch(env, fs=fs_env, nperseg=min(nperseg, env.size))
    return f, p, env


def match_peaks(f, p, targets, tol_pct=2.0, harmonics=3, min_snr=3.0, speed_hz=None):
    """Para cada frecuencia objetivo busca picos en sus primeros armonicos.
    SNR = amplitud del pico / mediana local del espectro.
    Un pico que coincide (dentro de la tolerancia) con un armonico de la
    velocidad de giro 1X se marca y NO cuenta como deteccion de rodamiento."""
    out = {}
    for name, f0 in targets.items():
        hits = []
        for k in range(1, harmonics + 1):
            fk = k * f0
            tol = fk * tol_pct / 100
            sel = (f >= fk - tol) & (f <= fk + tol)
            if not sel.any():
                continue
            idx = np.argmax(np.where(sel, p, -np.inf))
            noise = np.median(p[max(0, idx - 100):idx + 100])
            snr = float(p[idx] / noise) if noise > 0 else 0.0
            fpk = float(f[idx])
            on_1x = False
            if speed_hz and name != '1X':
                m = round(fpk / speed_hz)
                on_1x = bool(m >= 1 and abs(fpk - m * speed_hz) <= fk * tol_pct / 100)
            hits.append(dict(harmonic=k, freq_hz=fpk, snr=round(snr, 2),
                             coincides_with_1X=on_1x,
                             detected=bool(snr >= min_snr and not on_1x)))
        out[name] = dict(target_hz=float(f0), harmonics=hits,
                         detected=any(h['detected'] for h in hits))
    return out


def run_pipeline(x, fs, nlevel=6, speed_hz=None, bearing=None, env_fs=2500.0, nperseg=8192,
                 tol_pct=2.0, harmonics=3, min_snr=3.0, verbose=False):
    x = np.asarray(x, dtype=float).flatten()
    Kwav, Level_w, freq_w, c, info = fast_kurtogram(x, fs, nlevel, verbose)
    f, p, env = envelope_spectrum(x, fs, info['fc_hz'], info['bandwidth_hz'], env_fs, nperseg)
    result = dict(fs=fs, n_samples=int(x.size), kurtogram=info,
                  envelope_spectrum=dict(freq_hz=f.tolist(), power=p.tolist()))
    targets = {}
    if speed_hz:
        targets['1X'] = speed_hz
        if bearing:
            targets.update(bearing_frequencies(speed_hz, **bearing))
    if targets:
        result['targets_hz'] = {k: float(v) for k, v in targets.items()}
        result['matches'] = match_peaks(f, p, targets, tol_pct, harmonics, min_snr, speed_hz)
        result['fault_detected'] = any(m['detected'] for k, m in result['matches'].items() if k != '1X')
    result['_arrays'] = dict(Kwav=Kwav, Level_w=Level_w, freq_w=freq_w, c=c, env=env)
    return result

# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #

def load_signal(cfg):
    src = cfg['input']
    if src.get('format', 'labview_txt') == 'labview_txt':
        return np.genfromtxt(src['path'], skip_header=src.get('skip_header', 23), delimiter='\n')
    if src['format'] == 'npy':
        return np.load(src['path'])
    if src['format'] == 'csv':
        return np.loadtxt(src['path'], delimiter=',', usecols=src.get('column', 0), skiprows=src.get('skip_header', 0))
    raise ValueError('formato no soportado: %s' % src['format'])


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 1
    cfg = json.load(open(argv[1]))
    x = load_signal(cfg)
    fs = cfg['fs']
    speed_hz = cfg.get('speed_rpm', 0) / 60.0 or None
    kp = cfg.get('kurtogram', {}); ep = cfg.get('envelope', {})
    res = run_pipeline(x, fs, nlevel=kp.get('nlevel', 6), speed_hz=speed_hz,
                       bearing=cfg.get('bearing'), env_fs=ep.get('env_fs', 2500.0), nperseg=ep.get('nperseg', 8192),
                       tol_pct=ep.get('tolerance_pct', 2.0), harmonics=ep.get('harmonics', 3),
                       min_snr=ep.get('min_snr', 3.0), verbose=True)
    arrays = res.pop('_arrays')
    out = cfg.get('output', {}).get('results_json', 'results.json')
    json.dump(res, open(out, 'w'), indent=2)
    print('Resultados ->', out)
    if 'matches' in res:
        for k, m in res['matches'].items():
            print('  %-5s %8.2f Hz  %s' % (k, m['target_hz'], 'DETECTADO' if m['detected'] else '-'))
    if '--plot' in argv:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        prefix = cfg.get('output', {}).get('plot_prefix', 'kurtogram')
        plt.figure(figsize=(12, 4)); plt.imshow(arrays['Kwav'], aspect='auto',
            extent=[arrays['freq_w'][0], arrays['freq_w'][-1], len(arrays['Level_w']), 0])
        plt.yticks(range(len(arrays['Level_w'])), ['%.1f' % l for l in arrays['Level_w']])
        plt.xlabel('Hz'); plt.ylabel('nivel'); plt.colorbar(label='kurtosis'); plt.title('Fast Kurtogram')
        plt.tight_layout(); plt.savefig(prefix + '_kurtogram.png', dpi=120)
        f = np.array(res['envelope_spectrum']['freq_hz']); p = np.array(res['envelope_spectrum']['power'])
        plt.figure(figsize=(12, 4)); plt.semilogy(f, p); plt.xlim(0, 20 * (speed_hz or 50))
        for k, v in res.get('targets_hz', {}).items():
            plt.axvline(v, ls='--', lw=0.8, label=k)
        plt.legend(); plt.xlabel('Hz'); plt.title('Espectro de envolvente')
        plt.tight_layout(); plt.savefig(prefix + '_envelope.png', dpi=120)
        print('Figuras ->', prefix + '_kurtogram.png,', prefix + '_envelope.png')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
