import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

white_on_black_args = [
    path_effects.Stroke(linewidth=2, foreground="black"),
    path_effects.Normal(),
]

rc_params_dict = {
    "text.usetex": True,
    # "font.family": "Helvetica",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
}

## Anderson ###################################################################


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
    cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])

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


## Aubry-André ################################################################


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
