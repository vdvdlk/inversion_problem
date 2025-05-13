from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from analysis_functions import chi_array, load_arrays
from matplotlib.axes import Axes
from numpy.typing import NDArray

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


def countour_plot(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    ax_title: str,
    xlabel: str,
    ylabel: str,
    y_hline: float,
    x_vline: float,
    cbar_label: str,
    description: str,
):
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title(label=ax_title)

    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)

    ax.axhline(y=y_hline, color="red", linestyle="--")
    ax.axvline(x=x_vline, color="blue", linestyle="--")

    ticks = np.logspace(
        np.log10(z.min()),
        np.log10(z.max()),
        10,
    )

    contour = ax.contourf(
        x,
        y,
        z,
        levels=100,
        norm="log",
    )

    cbar = fig.colorbar(mappable=contour, ax=ax, label=cbar_label)
    cbar.set_ticks(ticks=ticks)
    cbar.set_ticklabels([f"{tick:.3f}" for tick in ticks])

    ax.text(
        # x=0.05,
        # y=0.95,
        x=0.01,
        y=0.99,
        # s=f"(a) {description}",
        s=description,
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=15,
        # color="red",
    )
    # ax.text(
    #     x=desc_pos[0],
    #     y=desc_pos[1],
    #     s=description,
    #     # color="white",
    #     transform=ax.transAxes,
    # )

    return fig, ax


def inset_plots(
    ax: Axes,
    x_1: NDArray[np.float_],
    y_1: NDArray[np.float_],
    yerr_1: NDArray[np.float_],
    x_vline_1: float,
    x_2: NDArray[np.float_],
    y_2: NDArray[np.float_],
    yerr_2: NDArray[np.float_],
    x_vline_2: float,
    inset_width: float,
    inset_height: float,
):
    """"""
    inset_1_bounds = [
        # 1.0 - inset_width,
        # 1.0 - inset_height,
        0.0,
        0.0,
        inset_width,
        inset_height,
    ]  # [x0, y0, width, height]
    inset_ax_1 = ax.inset_axes(bounds=inset_1_bounds)
    inset_ax_1.axvline(
        x=x_vline_1,
        color="blue",
        # alpha=0.5,
        linestyle="--",
    )
    inset_ax_1.errorbar(
        x=x_1,
        y=y_1,
        yerr=yerr_1,
        fmt="none",
        # fmt=".",
        # color="black",
        color="white",
        capsize=2.0,
    )
    inset_ax_1.tick_params(
        direction="in",
        top=False,
        bottom=False,
        left=False,
        right=True,
        labeltop=False,
        labelbottom=False,
        labelleft=False,
        labelright=True,
        labelcolor="white",
    )
    inset_ax_1.text(
        x=0.99,
        y=0.01,
        s="(b)",
        va="bottom",
        ha="right",
        transform=inset_ax_1.transAxes,
        fontsize=15,
        color="white",
    )  # .set_path_effects(white_on_black_args)
    inset_ax_1.set_facecolor("none")
    # y_1_ticks = [inset_ax_1.get_ylim()[0], inset_ax_1.get_ylim()[1]]
    y_1_ticks = [y_1.min(), inset_ax_1.get_ylim()[1]]
    y_1_labels = [f"{y_1_ticks[0]:.2f}", f"{y_1_ticks[1]:.2f}"]
    inset_ax_1.set_yticks(y_1_ticks)
    inset_ax_1.set_yticklabels(y_1_labels)

    inset_2_bounds = [
        1.0 - inset_width,
        0.0,
        inset_width,
        inset_height,
    ]  # [x0, y0, width, height]
    inset_ax_2 = ax.inset_axes(bounds=inset_2_bounds)
    inset_ax_2.axvline(
        x=x_vline_2,
        color="red",
        # alpha=0.5,
        linestyle="--",
    )
    inset_ax_2.errorbar(
        x=x_2,
        y=y_2,
        yerr=yerr_2,
        fmt="none",
        # color="black",
        color="white",
        # elinewidth=2.0,
        capsize=2.0,
    )
    inset_ax_2.tick_params(
        direction="in",
        top=False,
        bottom=False,
        left=True,
        right=False,
        # color="white",
        labeltop=False,
        labelbottom=False,
        labelleft=True,
        labelright=False,
        labelcolor="white",
    )
    inset_ax_2.tick_params(axis="x", labelcolor="white")
    inset_ax_2.text(
        x=1.00,
        y=0.01,
        s="(c)",
        va="bottom",
        ha="right",
        transform=inset_ax_2.transAxes,
        fontsize=15,
        color="white",
    )  # .set_path_effects(white_on_black_args)
    inset_ax_2.set_facecolor("none")
    # y_2_ticks = [inset_ax_2.get_ylim()[0], inset_ax_2.get_ylim()[1]]
    y_2_ticks = [y_2.min(), inset_ax_2.get_ylim()[1]]
    y_2_labels = [f"{y_2_ticks[0]:.2f}", f"{y_2_ticks[1]:.2f}"]
    inset_ax_2.set_yticks(y_2_ticks)
    inset_ax_2.set_yticklabels(y_2_labels)

    return inset_ax_1, inset_ax_2


def separate_inset_plots(
    x_1: NDArray[np.float_],
    y_1: NDArray[np.float_],
    yerr_1: NDArray[np.float_],
    x_vline_1: float,
    x_2: NDArray[np.float_],
    y_2: NDArray[np.float_],
    yerr_2: NDArray[np.float_],
    x_vline_2: float,
    x_1_label: str,
    x_2_label: str,
    ylabel: str,
):
    """"""
    fig = plt.figure()
    ax_1, ax_2 = fig.subplots(nrows=1, ncols=2, sharey=True)
    ax_1.set_xlabel(xlabel=x_1_label)
    ax_2.set_xlabel(xlabel=x_2_label)
    ax_1.set_ylabel(ylabel=ylabel)
    fig.subplots_adjust(wspace=0)

    ax_1.axvline(
        x=x_vline_1,
        color="blue",
        # alpha=0.5,
        linestyle="--",
    )
    ax_1.errorbar(
        x=x_1,
        y=y_1,
        yerr=yerr_1,
        fmt=".",
        # fmt=".",
        color="black",
        capsize=2.0,
    )
    ax_1.tick_params(
        direction="in",
        axis="both",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    ax_1.text(
        x=0.99,
        y=0.01,
        s="(a)",
        va="bottom",
        ha="right",
        transform=ax_1.transAxes,
        fontsize=15,
        # color="white",
    )
    # inset_ax_1.set_facecolor("none")

    ax_2.axvline(
        x=x_vline_2,
        color="red",
        # alpha=0.5,
        linestyle="--",
    )
    ax_2.errorbar(
        x=x_2,
        y=y_2,
        yerr=yerr_2,
        fmt=".",
        color="black",
        # elinewidth=2.0,
        capsize=2.0,
    )
    ax_2.tick_params(
        direction="in",
        axis="both",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    # inset_ax_2.tick_params(axis="x", labelcolor="white")
    ax_2.text(
        x=1.00,
        y=0.01,
        s="(b)",
        va="bottom",
        ha="right",
        transform=ax_2.transAxes,
        fontsize=15,
        # color="white",
    )
    # inset_ax_2.set_facecolor("none")

    return fig, (ax_1, ax_2)


if __name__ == "__main__":
    DIR = Path(__file__).parent
    ARRAY_DIR = DIR / "arrays"
    FIG_DIR = DIR.parent / "Figures" / "Anderson"
    # FIG_DIR.mkdir(parents=True, exist_ok=True)
    # print(f"{FIG_DIR=}")

    CA_DATA_ARRAYS_L100 = np.load(
        file=ARRAY_DIR / "dados_IP.npz",
    )
    ENERGIES: NDArray[np.float_] = CA_DATA_ARRAYS_L100["energies"]

    BARE_TTS_L100: NDArray[np.float_] = CA_DATA_ARRAYS_L100["tts"]
    BARE_LOGTT_L100: NDArray[np.float_] = np.log10(BARE_TTS_L100)
    CA_LOGTT_L100: NDArray[np.float_] = np.mean(BARE_LOGTT_L100, axis=-1)

    CA_SIGMAS: NDArray[np.float_] = CA_DATA_ARRAYS_L100["sigmas"]
    CA_GAMMAS: NDArray[np.float_] = CA_DATA_ARRAYS_L100["gammas"]

    ENERGIES, CA_SIGMAS, CA_GAMMAS, CA_LOGTT_L100 = load_arrays(
        array_filepath=ARRAY_DIR / "dados_IP.npz",
        data_type="ca",
    )

    INPUT_DATA_ARRAYS = np.load(
        file=ARRAY_DIR / "data_anderson_L100.npz",
    )
    INPUT_TT_L100: NDArray[np.float_] = INPUT_DATA_ARRAYS["tts"]
    INPUT_LOGTT_L100: NDArray[np.float_] = np.log10(INPUT_TT_L100)

    INPUT_SIGMAS = INPUT_DATA_ARRAYS["sigmas"]
    INPUT_GAMMAS = INPUT_DATA_ARRAYS["gammas"]

    # _, INPUT_SIGMAS, INPUT_GAMMAS, INPUT_LOGTT_L100 = load_arrays(
    #     array_filepath=ARRAY_DIR / "data_anderson_L100.npz",
    #     data_type="input",
    # )

    intervals = [
        (-2.0, 2.0),
        (0.0, 2.0),
        (1.0, 2.0),
    ]

    for interval in intervals:
        e_menos, e_mais = interval

        energy_dir = FIG_DIR / f"{e_menos:.1f}<E<{e_mais:.1f}"
        energy_dir.mkdir(parents=True, exist_ok=True)

        chis = chi_array(
            energies=ENERGIES,
            input_gamma=INPUT_LOGTT_L100,
            ca_gamma=CA_LOGTT_L100,
            e_menos=e_menos,
            e_mais=e_mais,
            # method="simpson",
        )
        n = chis.shape[-1]
        avg_chis = np.mean(chis, axis=-1)
        var_chis = np.var(chis, axis=-1, ddof=1)
        sdom_chi = np.sqrt(var_chis / n)

        chi_fig_2b, _ = separate_inset_plots(
            x_1=CA_GAMMAS[0, :],
            y_1=avg_chis[2, 4, :],
            yerr_1=sdom_chi[2, 4, :],
            x_vline_1=INPUT_GAMMAS[2],
            x_2=CA_SIGMAS[:, 0],
            y_2=avg_chis[2, :, 8],
            yerr_2=sdom_chi[2, :, 8],
            x_vline_2=INPUT_SIGMAS[2],
            x_1_label=r"$\gamma$",
            x_2_label=r"$W$",
            ylabel=r"$\overline{\chi}$",
        )
        # chi_fig_2b.savefig(
        #     fname=energy_dir / "chi_Ldep_W050_gamma_015.pdf",
        #     transparent=True,
        #     bbox_inches="tight",
        # )

        plt.show()
        plt.close("all")
