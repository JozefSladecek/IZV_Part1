#!/usr/bin/env python3
"""
IZV cast1 projektu 2025
Autor:

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene
na prednasce
"""
from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any

import os


def wave_inference_bad(
    x: NDArray[any], y: NDArray[any], sources: NDArray[any], wavelength: float
) -> NDArray[any]:
    """
    Referencni implementace, ktera je pomala a nevyuziva numpy efektivne;
    nezasahujte do ni!
    """
    k = 2 * np.pi / wavelength

    Z = np.zeros(x.shape + y.shape)
    for sx, sy in sources:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                R = np.sqrt((x[i] - sx) ** 2 + (y[j] - sy) ** 2)
                Z[j, i] += np.cos(k * R) / (1 + R)
    return Z


def wave_inference(
    x: NDArray[any], y: NDArray[any], sources: NDArray[any], wavelength: float
) -> NDArray[any]:
    k = 2 * np.pi / wavelength

    #Z = np.zeros(x.shape + y.shape)
    x,y = np.meshgrid(x,y)  # 1D -> 2D

    x = x[:, :, np.newaxis] # 2D -> 3D
    y = y[:, :, np.newaxis]

    # Broadcasting: sources[:,0] has shape (K,) where K = num_sources
    # NumPy expands: (200,200,1) and (K,) → both become (200,200,K)
    X = x - sources[:, 0] # x-differences from all sources
    Y = y - sources[:, 1]


    D = np.sqrt(X**2 + Y**2)

    Z = np.sum((np.cos(k * D) / (1+D)),axis=2)

    return Z


def plot_wave(
    Z: NDArray[any],
    x: NDArray[any],
    y: NDArray[any],
    show_figure: bool = False,
    save_path: str | None = None,
):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)

    #description of graph and range of axis
    ax.set_title("Vlnové pole")
    ax.set_xlabel('X pozice')
    ax.set_xlim(-10, 10)
    ax.set_ylabel('Y pozice')
    ax.set_ylim(-10, 10)

    #right bar
    img = ax.imshow(Z, extent=(-10, 10, -10, 10), origin='lower', cmap='viridis',vmin=-1, vmax=1)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Amplituda vlny')
    fig.show()

    save_path = os.path.join(os.getcwd(), "tmp_wave.png") #current folder
    fig.savefig(save_path)



    pass


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    pass


def download_data() -> Dict[str, List[Any]]:
    pass


if __name__ == "main":
    X = np.linspace(-10, 10, 200)
    Y = np.linspace(-10, 10, 200)
    #A = wave_inference_bad(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]), 2)
    B = wave_inference(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]),2)
    plot_wave(B, X, Y, show_figure=True)
    generate_sinus()
