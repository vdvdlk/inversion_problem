from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mod_analysis import chi_array, load_arrays
from mod_plotting import (
    countour_plot,
    inset_plots,
)

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


DIR = Path(__file__).parent
ARRAY_DIR = DIR / "arrays" / "anderson"
FIG_DIR = DIR / "figures" / "anderson"

ENERGIES, CA_SIGMAS, CA_GAMMAS, CA_LOGTT = load_arrays(
    array_filepath=ARRAY_DIR / "dados_IP.npz",
    data_type="ca",
)
CA_GAMMAS_2, CA_SIGMAS_2 = np.meshgrid(CA_GAMMAS, CA_SIGMAS)

_, INPUT_SIGMAS, INPUT_GAMMAS, INPUT_LOGTT = load_arrays(
    array_filepath=ARRAY_DIR / "data_anderson_L100.npz",
    data_type="input",
)

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
        input_gamma=INPUT_LOGTT[np.newaxis, np.newaxis, :, :, :],
        ca_gamma=CA_LOGTT[:, :, np.newaxis, np.newaxis, :],
        e_menos=e_menos,
        e_mais=e_mais,
    )
    n = chis.shape[-1]
    avg_chis = np.mean(chis, axis=-1)
    var_chis = np.var(chis, axis=-1, ddof=1)
    sdom_chi = np.sqrt(var_chis / n)

    chi_fig, chi_ax = countour_plot(
        x=CA_GAMMAS_2,
        y=CA_SIGMAS_2,
        z=avg_chis[:, :, 2],
        # ax_title=r"$ \log \mathrm{TT}$",
        ax_title="",
        xlabel=r"$\gamma$",
        ylabel=r"$W$",
        y_hline=INPUT_SIGMAS[2],
        x_vline=INPUT_GAMMAS[2],
        cbar_label=r"$\overline{\chi}$",
        description=rf"(a) ${e_menos} < E < {e_mais}$",
    )
    chi_insetax_hor, chi_insetax_ver = inset_plots(
        ax=chi_ax,
        x_1=CA_GAMMAS,
        y_1=avg_chis[4, :, 2],
        yerr_1=sdom_chi[4, :, 2],
        x_vline_1=INPUT_GAMMAS[2],
        x_2=CA_SIGMAS,
        y_2=avg_chis[:, 8, 2],
        yerr_2=sdom_chi[:, 8, 2],
        x_vline_2=INPUT_SIGMAS[2],
        inset_width=0.3,
        inset_height=0.3,
    )

    chi_fig.savefig(
        fname=energy_dir / "chi_insets_W050_gamma_015.pdf",
        transparent=True,
        bbox_inches="tight",
    )

    plt.show()
    plt.close("all")
