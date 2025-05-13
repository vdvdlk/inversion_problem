from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson, trapezoid


def new_shape_arrays(
    ca_array: NDArray,
    input_array: NDArray,
):
    """Retorna novos arrays de CA e de input com o mesmo shape"""
    ca_shape = ca_array.shape
    input_shape = input_array.shape

    new_ca_array = np.expand_dims(a=ca_array, axis=(0, -2))
    new_ca_array = np.repeat(
        a=new_ca_array,
        repeats=input_shape[0],
        axis=0,
    )
    new_ca_array = np.repeat(
        a=new_ca_array,
        repeats=input_shape[1],
        axis=-2,
    )

    new_input_array = np.expand_dims(a=input_array, axis=(1, -3))
    new_input_array = np.repeat(
        a=new_input_array,
        repeats=ca_shape[0],
        axis=1,
    )
    new_input_array = np.repeat(
        a=new_input_array,
        repeats=ca_shape[1],
        axis=-3,
    )

    # print(f"{ca_shape=}")
    # print(f"{input_shape=}")
    # print(f"{new_ca_array.shape=}")
    # print(f"{new_input_array.shape=}")

    return new_ca_array, new_input_array


def chi_array(
    energies: NDArray[np.float_],
    input_gamma: NDArray[np.float_],
    ca_gamma: NDArray[np.float_],
    e_menos: float,
    e_mais: float,
    # axis: int = -1,
    method: str = "trapezoid",
) -> NDArray[np.float_]:
    mask = (energies >= e_menos) & (energies <= e_mais)

    new_ca_gamma, new_input_gamma = new_shape_arrays(
        ca_array=ca_gamma,
        input_array=input_gamma,
    )
    integrand = (new_input_gamma - new_ca_gamma) ** 2

    # aplica máscara no último eixo (energias)
    integrand_masked = np.compress(mask, integrand, axis=-1)
    energies_masked = energies[mask]

    if method == "trapezoid":
        integral = trapezoid(
            x=energies_masked,
            y=integrand_masked,
            axis=-1,
        )
    elif method == "simpson":
        integral = simpson(
            x=energies_masked,
            y=integrand_masked,
            axis=-1,
        )
    elif method == "sum":
        delta_e = np.diff(energies_masked)[0]  # assume espaçamento uniforme
        integral = np.sum(integrand_masked, axis=-1) * delta_e
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return integral


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
        logtts: NDArray[np.float_] = np.mean(bare_logtts)
    else:
        raise ValueError(f"{data_type} is not a valid data_type")

    return energies, sigmas, gammas, logtts
