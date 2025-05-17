from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from analysis.arrays import load_arrays
from analysis.misfit_function import misfit_function
from analysis.plotting import countour_plot, separate_inset_plots

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

    chis = misfit_function(
        energies=ENERGIES,
        input_gamma=INPUT_LOGTT[np.newaxis, np.newaxis, :, :, :],
        ca_gamma=CA_LOGTT[:, :, np.newaxis, np.newaxis, :],
        e_menos=e_menos,
        e_mais=e_mais,
    )
    n = chis.shape[-1]
    avg_chis: NDArray[np.float_] = np.mean(chis, axis=-1)
    var_chis: NDArray[np.float_] = np.var(chis, axis=-1, ddof=1)
    sdom_chis = np.sqrt(var_chis / n)

    # print(f"{chis.shape=}")
    # print(f"{avg_chis.shape=}")

    chi_fig_0_2a, _ = countour_plot(
        x=CA_GAMMAS_2,
        y=CA_SIGMAS_2,
        z=avg_chis[:, :, 0],
        # ax_title=r"$ \log \mathrm{TT}$",
        ax_title="",
        xlabel=r"$\gamma$",
        ylabel=r"$W$",
        y_hline=INPUT_SIGMAS[0],
        x_vline=INPUT_GAMMAS[0],
        cbar_label=r"$\overline{\chi}$",
        description=rf"${e_menos} < E < {e_mais}$",
    )

    chi_fig_1_2a, _ = countour_plot(
        x=CA_GAMMAS_2,
        y=CA_SIGMAS_2,
        z=avg_chis[:, :, 1],
        # ax_title=r"$ \log \mathrm{TT}$",
        ax_title="",
        xlabel=r"$\gamma$",
        ylabel=r"$W$",
        y_hline=INPUT_SIGMAS[1],
        x_vline=INPUT_GAMMAS[1],
        cbar_label=r"$\overline{\chi}$",
        description=rf"${e_menos} < E < {e_mais}$",
    )

    chi_fig_2_2a, _ = countour_plot(
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
        description=rf"${e_menos} < E < {e_mais}$",
    )

    chi_fig_3_2a, _ = countour_plot(
        x=CA_GAMMAS_2,
        y=CA_SIGMAS_2,
        z=avg_chis[:, :, 3],
        # ax_title=r"$ \log \mathrm{TT}$",
        ax_title="",
        xlabel=r"$\gamma$",
        ylabel=r"$W$",
        y_hline=INPUT_SIGMAS[3],
        x_vline=INPUT_GAMMAS[3],
        cbar_label=r"$\overline{\chi}$",
        description=rf"${e_menos} < E < {e_mais}$",
    )

    chi_fig_4_2a, _ = countour_plot(
        x=CA_GAMMAS_2,
        y=CA_SIGMAS_2,
        z=avg_chis[:, :, 4],
        # ax_title=r"$ \log \mathrm{TT}$",
        ax_title="",
        xlabel=r"$\gamma$",
        ylabel=r"$W$",
        y_hline=INPUT_SIGMAS[4],
        x_vline=INPUT_GAMMAS[4],
        cbar_label=r"$\overline{\chi}$",
        description=rf"${e_menos} < E < {e_mais}$",
    )

    chi_fig_2b, _ = separate_inset_plots(
        x_1_label=r"$\gamma$",
        x_1=CA_GAMMAS,
        y_1=avg_chis[4, :, 2],
        yerr_1=sdom_chis[4, :, 2],
        x_vline_1=INPUT_GAMMAS[2],
        x_2_label=r"$W$",
        x_2=CA_SIGMAS,
        y_2=avg_chis[:, 8, 2],
        yerr_2=sdom_chis[:, 8, 2],
        x_vline_2=INPUT_SIGMAS[2],
        ylabel=r"$\overline{\chi}$",
    )

    chi_fig_0_2a.savefig(
        fname=energy_dir / "chi_0_W050_gamma_015.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    chi_fig_1_2a.savefig(
        fname=energy_dir / "chi_1_W050_gamma_015.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    chi_fig_2_2a.savefig(
        fname=energy_dir / "chi_2_W050_gamma_015.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    chi_fig_3_2a.savefig(
        fname=energy_dir / "chi_3_W050_gamma_015.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    chi_fig_4_2a.savefig(
        fname=energy_dir / "chi_4_W050_gamma_015.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    chi_fig_2b.set_size_inches(w=6.4 * 1.5, h=4.8)
    chi_fig_2b.savefig(
        fname=energy_dir / "chi_min_W050_gamma_015.pdf",
        transparent=True,
        bbox_inches="tight",
    )

    plt.show()
    plt.close("all")
