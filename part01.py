#!/usr/bin/env python3
"""
IZV cast1 projektu 2025
Autor: Jozef Sládeček (xsladej00)

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
    """
    Calculate wave amplitude from multiple sources using efficient NumPy operations.

    Parameters
    ----------
    x : NDArray
        X-axis coordinates
    y : NDArray
        Y-axis coordinates
    sources : NDArray
        Array of source positions
    wavelength : float
        Wave wavelength parameter

    Returns
    -------
    NDArray
        2D array of wave amplitudes, shape (len(y), len(x))
    """

    k = 2 * np.pi / wavelength

    x, y = np.meshgrid(x, y)  # 1D -> 2D

    x = x[:, :, np.newaxis] # 2D -> 3D
    y = y[:, :, np.newaxis]

    # Broadcasting: sources[:, 0] has shape (K,) where K = num_sources
    # NumPy expands: (200, 200, 1) and (K,) → both become (200, 200 ,K)
    X = x - sources[:, 0] # x-differences from all sources
    Y = y - sources[:, 1]


    D = np.sqrt(X**2 + Y**2)

    Z = np.sum((np.cos(k * D) / (1 + D)), axis=2)

    return Z


def plot_wave(
    Z: NDArray[any],
    x: NDArray[any],
    y: NDArray[any],
    show_figure: bool = False,
    save_path: str | None = None,
):
    """
    Visualize wave field as a 2D heatmap.

    Parameters
    ----------
    Z : NDArray
        2D array of wave amplitudes
    x : NDArray
        X-axis coordinates
    y : NDArray
        Y-axis coordinates
    show_figure : bool, optional
        If True, display the figure
    save_path : str | None, optional
        Path to save the figure, if provided
    """

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)

    # description of graph and range of axis
    ax.set_title("Vlnové pole")
    ax.set_xlabel('X pozice')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel('Y pozice')
    ax.set_ylim(y.min(), y.max())

    # right bar
    img = ax.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()),
                    origin='lower', cmap='viridis', vmin=-1, vmax=1)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Amplituda vlny')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()
    else:
        plt.close(fig)



def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generate visualization of sine and cosine functions.

    Creates two subplots:
    1. Both functions with filled area between them
    2. Maximum (colored by source) and minimum (dashed line)

    Parameters
    ----------
    show_figure : bool, optional
        If True, display the figure (default: False)
    save_path : str | None, optional
        Path to save the figure, if provided (default: None)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    # Configure axes appearance
    for ax in [ax1, ax2]:
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 4 * np.pi)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.7, alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Set LaTeX formatted x-axis labels
    x_ticks = np.linspace(0, 4*np.pi, 9)
    x_labels = [
        r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",
        r"$2\pi$", r"$\frac{5\pi}{2}$", r"$3\pi$", r"$\frac{7\pi}{2}$", r"$4\pi$"
    ]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, fontsize=7)

    # Generate data
    x = np.linspace(0, 4 * np.pi, 1000)
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    # Subplot 1: Both functions with filled area between them
    ax1.plot(x, sin_x, color='grey', linewidth=1.2)
    ax1.plot(x, cos_x, color='grey', linewidth=1.2)
    ax1.fill_between(x, sin_x, cos_x, color='green', alpha=0.2, interpolate=True)

    # Subplot 2: Maximum (colored by source) and minimum (dashed)
    y_max = np.maximum(sin_x, cos_x)
    y_min = np.minimum(sin_x, cos_x)

    cos_is_max = cos_x > sin_x

    ax2.plot(x, np.where(cos_is_max, y_max, np.nan), color='#ff7f0e', linewidth=1)  # Default matplotlib orange color
    ax2.plot(x, np.where(~cos_is_max, y_max, np.nan), color='#1f77b4', linewidth=1)  # Default matplotlib blue color
    ax2.plot(x, y_min, color='gray', linestyle='--', linewidth=1.2)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()
    else:
        plt.close()



def download_data() -> Dict[str, List[Any]]:
    """
    Download and parse meteorological station data from the specified URL.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'positions': list of station names
        - 'lats': list of latitudes (float)
        - 'longs': list of longitudes (float)
        - 'heights': list of heights above sea level (float)
    """

    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"
    response = requests.get(url)
    response.encoding = 'utf-8'
    html = response.text

    soup = BeautifulSoup(html, "html.parser")

    tables = soup.find_all('table')

    result = {
        "positions": [],
        "lats": [],
        "longs": [],
        "heights": []
    }

    numeric_rows = []
    for table in tables:
        for row in table.find_all('tr'):
            has_number = False
            cells = [cell.text.strip() for cell in row.find_all('td')]
            cleaned_cells = []
            for text in cells:
                cleaned = text.replace(",", ".").replace("°", "").strip()
                if not cleaned:
                    continue
                if '(' in text or ')' in cleaned:
                    continue
                cleaned_cells.append(cleaned)
                try:
                    float(cleaned)
                    has_number = True
                except ValueError:
                    continue
            if has_number:
                numeric_rows.append(cleaned_cells)

    for row in numeric_rows:
        if len(row) >= 4:
            result["positions"].append(row[0])
            try:
                result["lats"].append(float(row[1]))
            except ValueError:
                result["lats"].append(None)
            try:
                result["longs"].append(float(row[2]))
            except ValueError:
                result["longs"].append(None)
            try:
                result["heights"].append(float(row[3]))
            except ValueError:
                result["heights"].append(None)
    return result



if __name__ == "__main__":
    X = np.linspace(-10, 10, 200)
    Y = np.linspace(-10, 10, 200)
    A = wave_inference_bad(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]), 2)
    B = wave_inference(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]),2)
    plot_wave(B, X, Y, show_figure=True)
    generate_sinus(show_figure=True)
    download_data()

