"""
methods_v2.py - Perbaikan methods.py untuk revisi JESA 44740
Author: Kukuh Yudhistiro, 2026

Perubahan terhadap methods.py lama, semuanya menjawab poin reviewer:

  [R1-5, R2-5] Canny sekarang memakai citra yang SAMA dengan metode lain.
               Versi lama memberi Canny gray_u8 (tanpa histogram equalization)
               sementara metode lain memakai gray_f (dengan HE).
  [R1-1, R2-1] Canny menghasilkan peta kontinu, bukan biner. Versi lama
               menyimpan cv2.Canny/255 (biner) sehingga 99 ambang menghasilkan
               satu titik operasi, AP = 0, dan ODS pooled bisa melebihi OIS mean.
  [R1-5, R2-5] Border handling seragam. Versi lama memakai mode='constant'
               (zero padding) untuk GWi dan GWC, tetapi BORDER_REFLECT untuk
               A-GWi. Zero padding menciptakan tepi palsu di batas citra yang
               menekan presisi GWi dan GWC.
  [R1-2, R2-2] Semua parameter terpusat pada AGWiParams dan dapat dicetak.
  [R1-7, R2-7] Sakelar ablasi: adapt_freq, adapt_scale, l2_normalize,
               sigma_relation, rho_transform, rho_center, shuffle_rho.
  [R1-4, R2-4] rho_L diberi opsi transformasi CDF agar rentang efektifnya
               benar-benar mengisi [0, 1].

CATATAN PENTING (baca sebelum menjalankan apa pun):
  Dengan setelan lama (k_s = 25, pusat sigmoid = 0.5, f_min = 0.05),
  antara 79 dan 99 persen piksel menerima f0 dalam 1 persen dari f_min,
  karena rho_L pada BSDS500 bermedian sekitar 0.04 sampai 0.09 sedangkan
  sigmoid dipusatkan di 0.5. Adaptasi praktis tidak aktif. Jalankan
  diagnose_adaptation.py lebih dulu untuk memverifikasi ini pada data Anda.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import cv2

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*a, **k):
        def deco(f):
            return f
        return deco
    prange = range

try:
    from phasepack import phasecong
    HAS_PHASEPACK = True
except ImportError:
    HAS_PHASEPACK = False


ORIENTATIONS_8 = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

# Koefisien relasi bandwidth Gabor: sigma = COEF / f0
# 'paper'  : sqrt(ln2/2)/pi                      = 0.187391  (yang dipakai kode lama)
# 'octave' : 3*sqrt(ln2/2)/pi, b = 1 oktaf       = 0.562172  (relasi baku, cocok
#            dengan sigma = 0.56*lambda pada GWi statis)
SIGMA_COEF = {
    "paper": float(np.sqrt(np.log(2.0) / 2.0) / np.pi),
    "octave": float(3.0 * np.sqrt(np.log(2.0) / 2.0) / np.pi),
}


# ============================================================================
# Preprocessing (seragam untuk SEMUA metode)
# ============================================================================
def preprocess(image_path, use_he=True):
    """BT.601 grayscale, histogram equalization opsional, normalisasi [0,1].

    Returns
    -------
    gray_f  : float64 [0,1], masukan untuk Sobel, PC, GWC, GWi, A-GWi
    gray_u8 : uint8 [0,255] dari gray_f yang SAMA, masukan untuk Canny

    Versi lama mengembalikan gray_u8 sebelum HE, sehingga Canny menerima
    praproses berbeda dari metode lain. Di sini keduanya berasal dari citra
    yang sama.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    b, g, r = cv2.split(img)
    gray = (0.299 * r.astype(np.float64) +
            0.587 * g.astype(np.float64) +
            0.114 * b.astype(np.float64))
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    if use_he:
        gray_u8 = cv2.equalizeHist(gray_u8)
    return gray_u8.astype(np.float64) / 255.0, gray_u8


# ============================================================================
# Kernel Gabor statis (GWi, GWC)
# ============================================================================
def _gabor_grid(ksize, theta_deg):
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    th = np.deg2rad(theta_deg)
    x_t = x * np.cos(th) + y * np.sin(th)
    y_t = -x * np.sin(th) + y * np.cos(th)
    return x_t, y_t


def gabor_imaginary(ksize, wavelength, theta_deg, gamma=0.5, l2_normalize=False):
    sigma = 0.56 * wavelength
    x_t, y_t = _gabor_grid(ksize, theta_deg)
    gauss = np.exp(-0.5 * (x_t ** 2 + gamma ** 2 * y_t ** 2) / sigma ** 2)
    k = gauss * np.sin(2 * np.pi * x_t / wavelength)
    if l2_normalize:
        n = np.linalg.norm(k)
        if n > 0:
            k = k / n
    return k


def gabor_complex(ksize, wavelength, theta_deg, gamma=0.5, l2_normalize=False):
    sigma = 0.56 * wavelength
    x_t, y_t = _gabor_grid(ksize, theta_deg)
    gauss = np.exp(-0.5 * (x_t ** 2 + gamma ** 2 * y_t ** 2) / sigma ** 2)
    s = 2 * np.pi * x_t / wavelength
    kr, ki = gauss * np.cos(s), gauss * np.sin(s)
    if l2_normalize:
        nr, ni = np.linalg.norm(kr), np.linalg.norm(ki)
        if nr > 0:
            kr = kr / nr
        if ni > 0:
            ki = ki / ni
    return kr, ki


# ============================================================================
# Baseline
# ============================================================================
def run_canny_soft(gray_u8, n_levels=25, hi_lo=(20.0, 240.0), lo_ratio=0.4):
    """Canny dengan sapuan histeresis, menghasilkan peta kontinu.

    Untuk setiap threshold tinggi t dalam grid, jalankan cv2.Canny dengan
    (0.4t, t) dan akumulasi peta binernya. Nilai keluaran adalah fraksi
    ambang yang mempertahankan piksel tersebut, sehingga peta ini bersarang
    dan dapat disapu ulang oleh protokol Berkeley dengan 99 ambang.
    Ini memberi Canny kurva PR dan AP yang sah.
    """
    t0 = time.perf_counter()
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    his = np.linspace(hi_lo[0], hi_lo[1], n_levels)
    acc = np.zeros(gray_u8.shape, dtype=np.float64)
    for hi in his:
        lo = int(max(0, lo_ratio * hi))
        e = cv2.Canny(blurred, lo, int(hi))
        acc += (e > 0)
    acc /= float(n_levels)
    return acc, time.perf_counter() - t0


def run_canny_gm(gray_f, gaussian_sigma=1.4):
    """Magnitudo gradien gaya Canny sebelum NMS dan histeresis.

    Ini definisi yang dipakai pada Paper 1 (gwi-odps/src/baselines.py).
    Sertakan varian ini bila Anda ingin angka Paper 1 dan Paper 2 sebanding.
    Beri label eksplisit di naskah, jangan sebut "Canny" begitu saja.
    """
    t0 = time.perf_counter()
    u8 = (gray_f * 255.0).astype(np.uint8)
    sm = cv2.GaussianBlur(u8, (0, 0), gaussian_sigma)
    gx = cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2), time.perf_counter() - t0


def run_canny_binary(gray_u8):
    """Versi lama, disimpan hanya untuk mereproduksi angka naskah asli.
    JANGAN dipakai untuk hasil revisi: keluarannya biner sehingga AP = 0.
    """
    t0 = time.perf_counter()
    v = np.median(gray_u8)
    lo = int(max(0, 0.67 * v))
    hi = int(min(255, 1.33 * v))
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, lo, hi)
    return edges.astype(np.float64) / 255.0, time.perf_counter() - t0


def run_sobel(gray_f):
    t0 = time.perf_counter()
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2), time.perf_counter() - t0


def run_pc(gray_f, nscale=4, norient=6, min_wavelength=6, mult=2.1,
           sigma_onf=0.55, k=2.0):
    if not HAS_PHASEPACK:
        raise ImportError("phasepack not installed")
    t0 = time.perf_counter()
    M = phasecong(gray_f, nscale=nscale, norient=norient,
                  minWaveLength=min_wavelength, mult=mult,
                  sigmaOnf=sigma_onf, k=k)[0]
    return np.asarray(M, dtype=np.float64), time.perf_counter() - t0


def run_gwc(gray_f, ksize=7, wavelength=4.0, l2_normalize=False):
    """GWC dengan BORDER_REFLECT (versi lama memakai zero padding)."""
    t0 = time.perf_counter()
    best = None
    for th in ORIENTATIONS_8:
        kr, ki = gabor_complex(ksize, wavelength, th, l2_normalize=l2_normalize)
        rr = cv2.filter2D(gray_f, cv2.CV_64F, kr, borderType=cv2.BORDER_REFLECT)
        ri = cv2.filter2D(gray_f, cv2.CV_64F, ki, borderType=cv2.BORDER_REFLECT)
        m = np.sqrt(rr ** 2 + ri ** 2)
        best = m if best is None else np.maximum(best, m)
    return best, time.perf_counter() - t0


def run_gwi(gray_f, ksize=7, wavelength=4.0, l2_normalize=False):
    """GWi dengan BORDER_REFLECT (versi lama memakai zero padding)."""
    t0 = time.perf_counter()
    best = None
    for th in ORIENTATIONS_8:
        ki = gabor_imaginary(ksize, wavelength, th, l2_normalize=l2_normalize)
        ri = np.abs(cv2.filter2D(gray_f, cv2.CV_64F, ki,
                                 borderType=cv2.BORDER_REFLECT))
        best = ri if best is None else np.maximum(best, ri)
    return best, time.perf_counter() - t0


# ============================================================================
# Estimasi rho_L
# ============================================================================
def estimate_density_sobel(image_norm, sobel_ksize=5, blur_ksize=5):
    gx = cv2.Sobel(image_norm, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(image_norm, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    energy = np.sqrt(gx ** 2 + gy ** 2)
    rho = cv2.normalize(energy, None, 0, 1, cv2.NORM_MINMAX)
    rho = cv2.GaussianBlur(rho, (blur_ksize, blur_ksize), 0)
    return rho.astype(np.float64)


def estimate_density_variance(image_norm, win=5, blur_ksize=5):
    """Kontrol untuk ablasi A8: estimator non-gradien."""
    mean = cv2.boxFilter(image_norm, cv2.CV_64F, (win, win))
    mean_sq = cv2.boxFilter(image_norm ** 2, cv2.CV_64F, (win, win))
    var = np.maximum(mean_sq - mean ** 2, 0)
    rho = cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX)
    rho = cv2.GaussianBlur(rho, (blur_ksize, blur_ksize), 0)
    return rho.astype(np.float64)


def transform_rho(rho, mode="none", rng_seed=0):
    """Transformasi rho_L sebelum masuk sigmoid.

    'none'    : dipakai apa adanya (perilaku lama)
    'cdf'     : rank-normalisasi per citra ke [0,1] uniform. Membuat pusat
                sigmoid 0.5 benar-benar berada di median, sehingga adaptasi
                aktif. Ini perbaikan yang direkomendasikan.
    'shuffle' : permutasi spasial acak. Kontrol negatif untuk ablasi A7.
    """
    if mode == "none":
        return rho
    if mode == "cdf":
        flat = rho.ravel()
        order = np.argsort(flat, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(flat.size, dtype=np.float64)
        return (ranks / max(flat.size - 1, 1)).reshape(rho.shape)
    if mode == "shuffle":
        rng = np.random.default_rng(rng_seed)
        flat = rho.ravel().copy()
        rng.shuffle(flat)
        return flat.reshape(rho.shape)
    raise ValueError(f"Unknown rho_transform: {mode}")


# ============================================================================
# A-GWi
# ============================================================================
@dataclass
class AGWiParams:
    f_min: float = 0.05
    f_max: float = 0.45
    k_steepness: float = 25.0
    rho_center: float = 0.5        # BARU: pusat sigmoid, dulu terkunci 0.5
    n_orientations: int = 8
    aspect: float = 0.5
    kernel_size: int = 7
    sobel_ksize: int = 5
    density_blur: int = 5
    # sakelar ablasi
    adapt_freq: bool = True        # A3 jika False
    adapt_scale: bool = True       # A2 jika False
    sigma_relation: str = "paper"  # 'paper' atau 'octave'
    l2_normalize: bool = False     # A11 jika True
    f_static: float = 0.25         # f0 tetap saat adapt_freq = False
    sigma_static: float = 2.24     # sigma tetap saat adapt_scale = False

    def describe(self):
        return ", ".join(f"{k}={v}" for k, v in asdict(self).items())


@jit(nopython=True, cache=True, parallel=True)
def _agwi_svc(image_padded, rho_L, H, W, ksize, n_orient,
              k_s, f_min, f_max, rho_center, aspect,
              sigma_coef, adapt_freq, adapt_scale,
              f_static, sigma_static, l2_normalize):
    half_k = ksize // 2
    output = np.zeros((H, W))
    step = np.pi / n_orient
    for r in prange(H):
        for c in range(W):
            rho = rho_L[r, c]
            f_sig = 1.0 / (1.0 + np.exp(-k_s * (rho - rho_center)))
            if adapt_freq:
                f0 = f_min + (f_max - f_min) * f_sig
            else:
                f0 = f_static
            if adapt_scale:
                # skala mengikuti frekuensi adaptif melalui relasi bandwidth
                f_for_sigma = f_min + (f_max - f_min) * f_sig
                sigma_x = sigma_coef / f_for_sigma
            else:
                sigma_x = sigma_static
            sigma_y = sigma_x / aspect
            max_mag = 0.0
            for o in range(n_orient):
                theta = o * step
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                response = 0.0
                sq = 0.0
                for xi in range(ksize):
                    for yi in range(ksize):
                        x = xi - half_k + 0.0
                        y = yi - half_k + 0.0
                        xp = x * cos_t + y * sin_t
                        yp = -x * sin_t + y * cos_t
                        gauss = np.exp(-0.5 * (xp ** 2 / sigma_x ** 2
                                               + yp ** 2 / sigma_y ** 2))
                        kval = gauss * np.sin(2.0 * np.pi * f0 * xp)
                        response += image_padded[r + xi, c + yi] * kval
                        sq += kval * kval
                if l2_normalize and sq > 0.0:
                    response = response / np.sqrt(sq)
                m = abs(response)
                if m > max_mag:
                    max_mag = m
            output[r, c] = max_mag
    return output


def compute_f0_sigma_maps(rho_L, params):
    """Peta f0 dan sigma per piksel, untuk Figure baru (poin R1-15)."""
    f_sig = 1.0 / (1.0 + np.exp(-params.k_steepness *
                                (rho_L - params.rho_center)))
    f0 = params.f_min + (params.f_max - params.f_min) * f_sig
    coef = SIGMA_COEF[params.sigma_relation]
    sigma = coef / np.maximum(f0, 1e-9)
    if not params.adapt_freq:
        f0 = np.full_like(rho_L, params.f_static)
    if not params.adapt_scale:
        sigma = np.full_like(rho_L, params.sigma_static)
    return f0, sigma


def run_agwi(gray_f, params=None, rho_L=None, density_method="sobel",
             rho_transform_mode="none", rng_seed=0):
    if params is None:
        params = AGWiParams()
    H, W = gray_f.shape
    t0 = time.perf_counter()
    if rho_L is None:
        if density_method == "variance":
            rho_L = estimate_density_variance(gray_f, params.sobel_ksize,
                                              params.density_blur)
        else:
            rho_L = estimate_density_sobel(gray_f, params.sobel_ksize,
                                           params.density_blur)
    rho_L = transform_rho(rho_L, rho_transform_mode, rng_seed)
    half_k = params.kernel_size // 2
    padded = cv2.copyMakeBorder(gray_f.astype(np.float64),
                                half_k, half_k, half_k, half_k,
                                cv2.BORDER_REFLECT)
    mag = _agwi_svc(padded, np.ascontiguousarray(rho_L), H, W,
                    params.kernel_size, params.n_orientations,
                    params.k_steepness, params.f_min, params.f_max,
                    params.rho_center, params.aspect,
                    SIGMA_COEF[params.sigma_relation],
                    params.adapt_freq, params.adapt_scale,
                    params.f_static, params.sigma_static,
                    params.l2_normalize)
    return mag, time.perf_counter() - t0


def run_rho_as_edgemap(gray_f, params=None, density_method="sobel"):
    """Ablasi A9: rho_L langsung dievaluasi sebagai peta tepi.

    Ini kontrol paling penting. Jika rho_L sendiri mencetak ODS yang dekat
    dengan A-GWi, sebagian besar keunggulan berasal dari prior gradien Sobel,
    bukan dari kernel Gabor adaptif.
    """
    if params is None:
        params = AGWiParams()
    t0 = time.perf_counter()
    if density_method == "variance":
        rho = estimate_density_variance(gray_f, params.sobel_ksize,
                                        params.density_blur)
    else:
        rho = estimate_density_sobel(gray_f, params.sobel_ksize,
                                     params.density_blur)
    return rho, time.perf_counter() - t0


def warmup_agwi(params=None):
    if NUMBA_AVAILABLE:
        _ = run_agwi(np.random.rand(16, 16), params)
        return True
    return False


# ============================================================================
# Dispatcher
# ============================================================================
def run_method(method_name, gray_f, gray_u8, agwi_params=None,
               density_method="sobel", rho_transform_mode="none",
               rng_seed=0, l2_normalize_static=False):
    if method_name == "AGWi":
        return run_agwi(gray_f, agwi_params, density_method=density_method,
                        rho_transform_mode=rho_transform_mode,
                        rng_seed=rng_seed)
    if method_name == "RhoL":
        return run_rho_as_edgemap(gray_f, agwi_params, density_method)
    if method_name == "GWi":
        return run_gwi(gray_f, l2_normalize=l2_normalize_static)
    if method_name == "GWC":
        return run_gwc(gray_f, l2_normalize=l2_normalize_static)
    if method_name == "Canny":
        return run_canny_soft(gray_u8)
    if method_name == "CannyGM":
        return run_canny_gm(gray_f)
    if method_name == "CannyBinary":
        return run_canny_binary(gray_u8)
    if method_name == "Sobel":
        return run_sobel(gray_f)
    if method_name == "PC":
        return run_pc(gray_f)
    raise ValueError(f"Unknown method: {method_name}")
