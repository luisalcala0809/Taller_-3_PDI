"""Utilidades de filtros de Gabor para extraer vectores de textura.

Tarea 2 – Procesamiento Digital de Imágenes (Universidad de Antioquia).
Este módulo se usa tanto en el taller de filtros (Parte 1, Módulo E) como en la Parte 2.

Banco por defecto: 4 orientaciones (0°, 45°, 90°, 135°) × 3 longitudes de onda (4, 8 y 16 px).
Cada kernel de Gabor se construye con ``cv2.getGaborKernel`` y se le aplica:
  * ancho de banda fijo: sigma = 0.56·lambda (≈ 1 octava), de modo que cada kernel contiene
    el mismo número de ciclos dentro de su envolvente gaussiana;
  * relación de aspecto gamma = 0.5 (envolvente elíptica, alargada a lo largo de las franjas);
  * fase psi = 0 (kernel par, detector de "franjas");
  * media cero (se resta la media del kernel para ignorar el brillo medio) y normalización
    por la suma de valores absolutos (las respuestas de distintas escalas son comparables).
"""
from functools import lru_cache

import cv2
import numpy as np

ORIENTACIONES_GRADOS = (0, 45, 90, 135)
LONGITUDES_ONDA = (4, 8, 16)
GAMMA = 0.5
PSI = 0.0
ANCHO_BANDA = 0.56          # sigma = ANCHO_BANDA * lambda


def kernel_gabor(longitud_onda, theta_grados, gamma=GAMMA, psi=PSI, ancho_banda=ANCHO_BANDA):
    """Kernel de Gabor 2D (float64) de media cero y norma L1 unitaria.

    Parameters
    ----------
    longitud_onda : float
        Longitud de onda de la sinusoide (píxeles por ciclo).
    theta_grados : float
        Orientación del kernel en grados (0° detecta franjas verticales).
    gamma, psi, ancho_banda : float
        Relación de aspecto, fase y sigma/lambda.
    """
    sigma = ancho_banda * longitud_onda
    mitad = int(np.ceil(3 * sigma))
    k = cv2.getGaborKernel((2 * mitad + 1, 2 * mitad + 1), sigma, np.deg2rad(theta_grados),
                           longitud_onda, gamma, psi, ktype=cv2.CV_64F)
    k -= k.mean()                       # media cero: insensible al brillo medio
    return k / np.abs(k).sum()          # energía comparable entre escalas


@lru_cache(maxsize=None)
def _banco_cache(longitudes, orientaciones, gamma, psi, ancho_banda):
    return tuple((lam, th, kernel_gabor(lam, th, gamma, psi, ancho_banda))
                 for lam in longitudes for th in orientaciones)


def construir_banco(longitudes_onda=LONGITUDES_ONDA, orientaciones=ORIENTACIONES_GRADOS,
                    gamma=GAMMA, psi=PSI, ancho_banda=ANCHO_BANDA):
    """Devuelve una lista de tuplas (lambda, theta_grados, kernel), ordenada por lambda y luego por theta."""
    return list(_banco_cache(tuple(longitudes_onda), tuple(orientaciones), gamma, psi, ancho_banda))


def _a_float01(img):
    """Convierte a gris float64 en [0,1] (acepta uint8 o float; si es color, lo pasa a gris)."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8) if img.dtype != np.uint8 else img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def respuestas_gabor(img, banco=None):
    """Respuestas (float64, mismo tamaño que img) de cada kernel del banco; lista en el orden del banco."""
    banco = banco or construir_banco()
    f = _a_float01(img)
    return [cv2.filter2D(f, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT) for _, _, k in banco]


def vector_gabor(img, banco=None):
    """Vector de textura de Gabor: media y desviación estándar de la energía (respuesta²) de cada filtro.

    Parameters
    ----------
    img : ndarray
        Imagen en gris (uint8 o float). Si es uint8 se normaliza a [0,1].
    banco : list, optional
        Banco de ``construir_banco`` (por defecto 3 longitudes de onda × 4 orientaciones).

    Returns
    -------
    ndarray de longitud 2·len(banco) (24 con el banco por defecto), ordenado por
    (lambda, theta, [media, desviación]) y aplanado: es decir,
    ``v.reshape(n_lambdas, n_thetas, 2)[i, j] = (media, std)`` de la energía del kernel (lambda_i, theta_j).
    """
    banco = banco or construir_banco()
    caracteristicas = []
    for r in respuestas_gabor(img, banco):
        energia = r ** 2
        caracteristicas.extend([energia.mean(), energia.std()])
    return np.asarray(caracteristicas, dtype=np.float64)


def vector_gabor_invariante(img, banco=None, modo="promedio"):
    """Descriptor de Gabor invariante a rotaciones de 90° (y a cualquier permutación de las orientaciones).

    modo='promedio' : para cada lambda, media y desviación **a través de las orientaciones** de la
                      media de energía y de la desviación de energía -> 4 números por lambda
                      (12 con el banco por defecto). Invariante a cualquier permutación de theta.
    modo='alineado' : desplaza circularmente las orientaciones hasta que la de mayor energía total
                      (suma sobre lambdas) quede primera; conserva las 24 componentes. Es invariante a
                      rotaciones de 90° cuando la orientación dominante es única.
    """
    banco = banco or construir_banco()
    lambdas = sorted({lam for lam, _, _ in banco})
    thetas = sorted({th for _, th, _ in banco})
    v = vector_gabor(img, banco).reshape(len(lambdas), len(thetas), 2)       # (lambda, theta, [media, std])
    if modo == "promedio":
        return np.concatenate([[v[i, :, 0].mean(), v[i, :, 0].std(), v[i, :, 1].mean(), v[i, :, 1].std()]
                               for i in range(len(lambdas))])
    if modo == "alineado":
        dominante = int(np.argmax(v[:, :, 0].sum(axis=0)))
        return np.roll(v, -dominante, axis=1).ravel()
    raise ValueError("modo debe ser 'promedio' o 'alineado'")


def nombres_caracteristicas(banco=None):
    """Etiquetas legibles del vector de ``vector_gabor`` (mismo orden)."""
    banco = banco or construir_banco()
    return [f"λ={lam:g} θ={th:g}° {stat}" for lam, th, _ in banco for stat in ("media", "std")]
