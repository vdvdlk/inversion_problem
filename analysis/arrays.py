"""Módulo com funções usadas nas análises"""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from uncertainties import UFloat  # noqa: F401
from uncertainties import unumpy as unp  # noqa: F401


def load_arrays(
    array_filepath: Path,
    data_type: Literal["ca", "input"],
):
    data: NDArray[np.float_] = np.load(file=array_filepath)

    energies: NDArray[np.float_] = data["energies"]
    sigmas: NDArray[np.float_] = data["sigmas"]
    gammas: NDArray[np.float_] = data["gammas"]

    bare_tts: NDArray[np.float_] = data["tts"]
    bare_logtts = np.log10(bare_tts)
    if data_type == "ca":
        logtts: NDArray[np.float_] = np.mean(bare_logtts, axis=-1)
    elif data_type == "input":
        logtts: NDArray[np.float_] = bare_logtts
    else:
        raise ValueError(f"{data_type} is not a valid data_type")
    # print(f"{bare_logtts.shape=}, {logtts.shape=}, {data_type=}")

    return energies, sigmas, gammas, logtts


def normalize(
    array: NDArray,
    sdom_array: NDArray,
):
    uarray = unp.uarray(nominal_values=array, std_devs=sdom_array)

    norm_uarray = (uarray - uarray.min()) / (uarray.max() - uarray.min())

    norm_array: NDArray[np.float_] = unp.nominal_values(norm_uarray)
    norm_sdom_uarray: NDArray[np.float_] = unp.std_devs(norm_uarray)

    return norm_array, norm_sdom_uarray
