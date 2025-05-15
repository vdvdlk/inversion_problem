from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mod_analysis import chi_array
from mod_plotting import chi_plot

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

if __name__ == "__main__":
    DIR = Path(__file__).parent
    ARRAY_DIR = DIR / "arrays" / "aubry_andre"
    FIG_DIR = DIR / "figures" / "aubry_andre"
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
            input_gamma=INPUT_LOGTT[:, np.newaxis, :, :],
            ca_gamma=CA_LOGTTS[:, :, np.newaxis, :],
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

        fig_2, ax_2 = chi_plot(
            x=CA_GAMMAS[1, :],
            y=avg_chis[1, :],
            yerr=sdom_chi[1, :],
            x_vline=INPUT_GAMMAS[1],
            xlabel=r"$\gamma$",
            ylabel=r"$\overline{\chi}$",
            description=rf"${e_menos} < E < {e_mais}$, $g_\text{{AA}} = {INPUT_G_AA_S[1]}$",
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

        # fig_1.savefig(
        #     fname=energy_dir / f"chi_gaa_{INPUT_G_AA_S[0]}.pdf",
        #     transparent=True,
        #     bbox_inches="tight",
        # )
        # fig_2.savefig(
        #     fname=energy_dir / f"chi_gaa_{INPUT_G_AA_S[1]}.pdf",
        #     transparent=True,
        #     bbox_inches="tight",
        # )
        # fig_3.savefig(
        #     fname=energy_dir / f"chi_gaa_{INPUT_G_AA_S[2]}.pdf",
        #     transparent=True,
        #     bbox_inches="tight",
        # )

        # plt.show()
        plt.close("all")
