from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson, trapezoid

# from matplotlib.figure import Figure

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

white_on_black_args = [
    path_effects.Stroke(linewidth=2, foreground="black"),
    path_effects.Normal(),
]


def new_shape_arrays(
    ca_array: NDArray,
    input_array: NDArray,
):
    """Retorna novos arrays de CA e de input com o mesmo shape"""
    ca_shape = ca_array.shape
    input_shape = input_array.shape

    new_ca_array = np.expand_dims(a=ca_array, axis=2)
    new_ca_array = np.repeat(
        a=new_ca_array,
        repeats=input_shape[1],
        axis=2,
    )

    new_input_array = np.expand_dims(a=input_array, axis=1)
    new_input_array = np.repeat(
        a=new_input_array,
        repeats=ca_shape[1],
        axis=1,
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


def chi_plot(
    x: NDArray,
    y: NDArray,
    yerr: NDArray,
    x_vline: float,
    ax_title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    description: str = "",
):
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title(label=ax_title)

    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)

    ax.axvline(x=x_vline, color="blue", linestyle="--")

    ax.errorbar(
        x=x,
        y=y,
        yerr=yerr,
        fmt=".",
        capsize=2.0,
    )

    ax.text(
        x=0.99,
        y=0.01,
        # s=f"(a) {description}",
        s=description,
        va="bottom",
        ha="right",
        transform=ax.transAxes,
        fontsize=15,
        # color="red",
    )

    return fig, ax


if __name__ == "__main__":
    DIR = Path(__file__).parent
    ARRAY_DIR = DIR / "arrays"
    FIG_DIR = DIR.parent / "Figures" / r"Aubry-André"
    # FIG_DIR.mkdir(parents=True, exist_ok=True)
    # print(f"{FIG_DIR=}")

    CA_DATA_ARRAYS = np.load(
        file=ARRAY_DIR / "AA_results.npz",
    )
    ENERGIES: NDArray[np.float_] = CA_DATA_ARRAYS["energies"]

    C = np.log10(np.e)
    CA_LOGTTS: NDArray[np.float_] = C * CA_DATA_ARRAYS["logtts"]
    CA_SDOM_LOGTTS: NDArray[np.float_] = C * CA_DATA_ARRAYS["sdom_logtts"]
    print(f"{CA_LOGTTS.shape=}")

    CA_G_AA_S: NDArray[np.float_] = CA_DATA_ARRAYS["g_aa_s"]
    CA_GAMMAS: NDArray[np.float_] = CA_DATA_ARRAYS["gammas"]

    INPUT_DATA_ARRAYS = np.load(
        file=ARRAY_DIR / "data_aah.npz",
    )
    INPUT_LOGTT: NDArray[np.float_] = C * INPUT_DATA_ARRAYS["logtts"]
    print(f"{INPUT_LOGTT.shape=}")

    INPUT_G_AA_S = INPUT_DATA_ARRAYS["g_aa_s"]
    INPUT_GAMMAS = INPUT_DATA_ARRAYS["gammas"]
    print(f"{INPUT_G_AA_S=}")
    print(f"{INPUT_GAMMAS=}")

    INTERVALS = [
        (-2.0, 2.0),
        (0.0, 2.0),
        (1.0, 2.0),
    ]

    for interval in INTERVALS:
        e_menos, e_mais = interval

        energy_dir = FIG_DIR / f"{e_menos:.1f}<E<{e_mais:.1f}"
        energy_dir.mkdir(parents=True, exist_ok=True)

        chis = chi_array(
            energies=ENERGIES,
            input_gamma=INPUT_LOGTT,
            ca_gamma=CA_LOGTTS,
            e_menos=e_menos,
            e_mais=e_mais,
            # method="simpson",
        )
        n = chis.shape[-1]
        avg_chis = np.mean(chis, axis=-1)
        var_chis = np.var(chis, axis=-1, ddof=1)
        sdom_chi = np.sqrt(var_chis / n)

        # print(f"{CA_GAMMAS.shape=}")
        # print(f"{avg_chis.shape=}")

        fig_1, ax_1 = chi_plot(
            x=CA_GAMMAS[0, :],
            y=avg_chis[0, :],
            yerr=sdom_chi[0, :],
            x_vline=INPUT_GAMMAS[0],
            xlabel=r"$\gamma$",
            ylabel=r"$\overline{\chi}$",
            description=rf"${e_menos} < E < {e_mais}$, $g_\text{{AA}} = {INPUT_G_AA_S[0]}$",
        )
        fig_1.savefig(
            fname=energy_dir / f"chi_gaa_{INPUT_G_AA_S[0]}.pdf",
            transparent=True,
            bbox_inches="tight",
        )

        fig_2, ax_2 = chi_plot(
            x=CA_GAMMAS[1, :],
            y=avg_chis[1, :],
            yerr=sdom_chi[1, :],
            x_vline=INPUT_GAMMAS[1],
            xlabel=r"$\gamma$",
            ylabel=r"$\overline{\chi}$",
            description=rf"${e_menos} < E < {e_mais}$, $g_\text{{AA}} = {INPUT_G_AA_S[1]}$",
        )
        fig_2.savefig(
            fname=energy_dir / f"chi_gaa_{INPUT_G_AA_S[1]}.pdf",
            transparent=True,
            bbox_inches="tight",
        )

        fig_3, ax_3 = chi_plot(
            x=CA_GAMMAS[2, :],
            y=avg_chis[2, :],
            yerr=sdom_chi[2, :],
            x_vline=INPUT_GAMMAS[2],
            xlabel=r"$\gamma$",
            ylabel=r"$\overline{\chi}$",
            description=rf"${e_menos} < E < {e_mais}$, $g_\text{{AA}} = {INPUT_G_AA_S[2]}$",
        )
        fig_3.savefig(
            fname=energy_dir / f"chi_gaa_{INPUT_G_AA_S[2]}.pdf",
            transparent=True,
            bbox_inches="tight",
        )

        # plt.close("all")
        # plt.show()
