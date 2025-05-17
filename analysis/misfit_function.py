"""Módulo com funções usadas nas análises"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson, trapezoid


def misfit_function(
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
    masked_integrand = np.compress(mask, integrand, axis=axis)
    masked_energies = energies[mask]

    if method == "trapezoid":
        integral = trapezoid(
            x=masked_energies,
            y=masked_integrand,
            axis=axis,
        )
    elif method == "simpson":
        integral = simpson(
            x=masked_energies,
            y=masked_integrand,
            axis=axis,
        )
    elif method == "sum":
        delta_e = np.diff(masked_energies)[0]  # assume espaçamento uniforme
        integral = np.sum(masked_integrand, axis=axis) * delta_e
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return integral
