from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mod_analysis import chi_array, load_arrays

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


DIR = Path(__file__).parent
ARRAY_DIR = DIR / "arrays" / "anderson"
FIG_DIR = DIR / "figures" / "anderson"

_, CA_SIGMAS_L100, CA_GAMMAS_L100, CA_LOGTT_L100 = load_arrays(
    array_filepath=ARRAY_DIR / "dados_IP.npz",
    data_type="ca",
)
ENERGIES, CA_SIGMAS_L200_W050, CA_GAMMAS_L200_W050, CA_LOGTT_L200_W050 = load_arrays(
    array_filepath=ARRAY_DIR / "data_anderson_ca_L200_W050.npz",
    data_type="ca",
)
ENERGIES, CA_SIGMAS_L200_GAMMA015, CA_GAMMAS_L200_GAMMA015, CA_LOGTT_L200_GAMMA015 = (
    load_arrays(
        array_filepath=ARRAY_DIR / "data_anderson_ca_L200_gamma015.npz",
        data_type="ca",
    )
)


_, INPUT_SIGMAS_L100, INPUT_GAMMAS_L100, INPUT_LOGTT_L100 = load_arrays(
    array_filepath=ARRAY_DIR / "data_anderson_L100.npz",
    data_type="input",
)

_, INPUT_SIGMAS_L200, INPUT_GAMMAS_L200, INPUT_LOGTT_L200 = load_arrays(
    array_filepath=ARRAY_DIR / "data_anderson_L200.npz",
    data_type="input",
)

intervals = [
    (-2.0, 2.0),
    (0.0, 2.0),
    (1.0, 2.0),
    (-2.0, -1.0),
    (-2.0, 0.0),
]

# print(f"{INPUT_LOGTT_L100.shape=}")
# print(f"{CA_LOGTT_L100.shape=}")

# print(INPUT_LOGTT_L100 - CA_LOGTT_L100)

for interval in intervals:
    e_menos, e_mais = interval

    energy_dir = FIG_DIR / f"{e_menos:.1f}<E<{e_mais:.1f}"
    energy_dir.mkdir(parents=True, exist_ok=True)

    chis_l100 = chi_array(
        energies=ENERGIES,
        input_gamma=INPUT_LOGTT_L100[np.newaxis, np.newaxis, :, :, :],
        ca_gamma=CA_LOGTT_L100[:, :, np.newaxis, np.newaxis, :],
        e_menos=e_menos,
        e_mais=e_mais,
    )
    # print(f"{chis_l100.shape=}")
    n_l100 = chis_l100.shape[-1]
    avg_chis_l100 = np.mean(chis_l100, axis=-1)
    var_chis_l100 = np.var(chis_l100, axis=-1, ddof=1)
    sdom_chis_l100 = np.sqrt(var_chis_l100 / n_l100)

    chis_l200_w050 = chi_array(
        energies=ENERGIES,
        input_gamma=INPUT_LOGTT_L200[np.newaxis, np.newaxis, :, :, :],
        ca_gamma=CA_LOGTT_L200_W050[:, :, np.newaxis, np.newaxis, :],
        e_menos=e_menos,
        e_mais=e_mais,
    )
    # print(f"{chis_l200_w050.shape=}")
    n_l200_w050 = chis_l200_w050.shape[-1]
    avg_chis_l200_w050 = np.mean(chis_l200_w050, axis=-1)
    var_chis_l200_w050 = np.var(chis_l200_w050, axis=-1, ddof=1)
    sdom_chis_l200_w050 = np.sqrt(var_chis_l200_w050 / n_l200_w050)

    chis_l200_gamma015 = chi_array(
        energies=ENERGIES,
        input_gamma=INPUT_LOGTT_L200[np.newaxis, np.newaxis, :, :, :],
        ca_gamma=CA_LOGTT_L200_GAMMA015[:, :, np.newaxis, np.newaxis, :],
        e_menos=e_menos,
        e_mais=e_mais,
    )
    # print(f"{chis_l200_gamma015.shape=}")
    n_l200_gamma015 = chis_l200_gamma015.shape[-1]
    avg_chis_l200_gamma015 = np.mean(chis_l200_gamma015, axis=-1)
    var_chis_l200_gamma015 = np.var(chis_l200_gamma015, axis=-1, ddof=1)
    sdom_chis_l200_gamma015 = np.sqrt(var_chis_l200_gamma015 / n_l200_gamma015)

    # print(f"{avg_chis_l100.shape=}")
    # print(f"{avg_chis_l200_gamma015.shape=}")
    # print(f"{CA_GAMMAS_L200_gamma015.shape=}")

    #########################################

    fig = plt.figure()
    axes = fig.subplots(nrows=1, ncols=2, sharey=True)
    fig.subplots_adjust(wspace=0)

    color_l100 = "green"
    fmt_l100 = "."
    color_l200 = "purple"
    fmt_l200 = "s"

    axes[0].set_xlabel(xlabel=r"$\gamma$")
    axes[0].set_ylabel(ylabel=r"$\overline{\chi}$")
    axes[0].axvline(
        x=INPUT_GAMMAS_L100[2],
        color="blue",
        # alpha=0.5,
        linestyle="--",
    )
    axes[0].errorbar(
        x=CA_GAMMAS_L100,
        y=avg_chis_l100[4, :, 2],
        yerr=sdom_chis_l100[4, :, 2],
        fmt=fmt_l100,
        color=color_l100,
        capsize=2.0,
        label=r"$L = 100$",
    )
    axes[0].errorbar(
        x=CA_GAMMAS_L200_W050,
        y=avg_chis_l200_w050[0, :, 2],
        yerr=sdom_chis_l200_w050[0, :, 2],
        fmt=fmt_l200,
        markersize=3.0,
        color=color_l200,
        capsize=2.0,
        label=r"$L = 200$",
    )
    axes[0].tick_params(
        direction="in",
        axis="both",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    axes[0].text(
        x=0.99,
        y=0.01,
        s="(a)",
        va="bottom",
        ha="right",
        transform=axes[0].transAxes,
        fontsize=15,
        # color="white",
    )
    axes[0].legend()

    axes[1].set_xlabel(xlabel=r"$W$")
    axes[1].axvline(
        x=INPUT_SIGMAS_L100[2],
        color="red",
        # alpha=0.5,
        linestyle="--",
    )
    axes[1].errorbar(
        x=CA_SIGMAS_L100,
        y=avg_chis_l100[:, 8, 2],
        yerr=sdom_chis_l100[:, 8, 2],
        fmt=fmt_l100,
        color=color_l100,
        label=r"$L = 100$",
        # elinewidth=2.0,
        capsize=2.0,
    )
    axes[1].errorbar(
        x=CA_SIGMAS_L200_GAMMA015,
        y=avg_chis_l200_gamma015[:, 0, 2],
        yerr=sdom_chis_l200_gamma015[:, 0, 2],
        fmt=fmt_l200,
        markersize=3.0,
        color=color_l200,
        label=r"$L = 200$",
        capsize=2.0,
    )
    axes[1].tick_params(
        direction="in",
        axis="both",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    # inset_ax_2.tick_params(axis="x", labelcolor="white")
    axes[1].text(
        x=1.00,
        y=0.01,
        s="(b)",
        va="bottom",
        ha="right",
        transform=axes[1].transAxes,
        fontsize=15,
        # color="white",
    )
    # inset_ax_2.set_facecolor("none")
    # axes[1].legend()
    axes[1].text(
        x=0.45,
        y=0.95,
        horizontalalignment="right",
        verticalalignment="top",
        s=rf"${e_menos} < E < {e_mais}$",
        # color="white",
        transform=axes[1].transAxes,
    )

    #######################################

    fig.set_size_inches(w=6.4 * 1.5, h=4.8)
    fig.savefig(
        fname=energy_dir / "chi_Ldep_W050_gamma_015.pdf",  # type: ignore
        transparent=True,
        bbox_inches="tight",
    )

    plt.show()
    plt.close("all")
