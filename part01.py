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

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    pass


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    # Configure axes appearance
    for ax in [ax1, ax2]:
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 4 * np.pi)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Set LaTeX formatted x-axis labels
    x_ticks = np.linspace(0, 4*np.pi, 9)
    x_labels = [
        r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",
        r"$2\pi$", r"$\frac{5\pi}{2}$", r"$3\pi$", r"$\frac{7\pi}{2}$", r"$4\pi$"
    ]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels,fontsize=7)

    # Generate data
    x = np.linspace(0, 4 * np.pi,1000)
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    # Subplot 1: Both functions with filled area between them
    ax1.plot(x, sin_x,color='grey')
    ax1.plot(x, cos_x,color='grey')
    ax1.fill_between(x, sin_x, cos_x, where=cos_x > sin_x, color='green', alpha=0.2,interpolate=True)
    ax1.fill_between(x, sin_x, cos_x, where=cos_x < sin_x, color='green', alpha=0.2,interpolate=True)

    # Subplot 2: Maximum (colored by source) and minimum (dashed)
    y_max = np.maximum(sin_x, cos_x)
    y_min = np.minimum(sin_x, cos_x)

    cos_is_max = cos_x > sin_x

    ax2.plot(x, np.where(cos_is_max, y_max, np.nan), color='#ff7f0e')
    ax2.plot(x, np.where(~cos_is_max, y_max, np.nan), color='#1f77b4') # default color
    ax2.plot(x, y_min, color='gray', linestyle='--')

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()
    else:
        plt.close()
    pass


def download_data() -> Dict[str, List[Any]]:
    pass


if __name__ == "__main__":
    X = np.linspace(-10, 10, 200)
    Y = np.linspace(-10, 10, 200)
    #A = wave_inference_bad(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]), 2)
    B = wave_inference(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]),2)
    plot_wave(B, X, Y, show_figure=True)
    generate_sinus(show_figure=True)

