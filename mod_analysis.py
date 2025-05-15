from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson, trapezoid


def chi_array(
    energies: NDArray[np.float_],
    input_gamma: NDArray[np.float_],
    ca_gamma: NDArray[np.float_],
    e_menos: float,
    e_mais: float,
    axis: int = -1,
    method: Literal["trapezoid", "simpson", "sum"] = "trapezoid",
) -> NDArray[np.float_]:
    if e_menos >= e_mais:
        raise ValueError(f"{e_mais=} has to be greater than {e_menos=}")
    mask = (energies >= e_menos) & (energies <= e_mais)

    integrand = (input_gamma - ca_gamma) ** 2

    # Aplica a máscara ao último eixo (energias)
    integrand_masked = np.compress(mask, integrand, axis=axis)
    energies_masked = energies[mask]

    if method == "trapezoid":
        integral = trapezoid(
            x=energies_masked,
            y=integrand_masked,
            axis=axis,
        )
    elif method == "simpson":
        integral = simpson(
            x=energies_masked,
            y=integrand_masked,
            axis=axis,
        )
    elif method == "sum":
        delta_e = np.diff(energies_masked)[0]  # assume espaçamento uniforme
        integral = np.sum(integrand_masked, axis=axis) * delta_e
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
        logtts: NDArray[np.float_] = bare_logtts
    else:
        raise ValueError(f"{data_type} is not a valid data_type")
    # print(f"{bare_logtts.shape=}, {logtts.shape=}, {data_type=}")

    return energies, sigmas, gammas, logtts
