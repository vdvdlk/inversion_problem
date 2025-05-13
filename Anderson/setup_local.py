"""Setup the AA chain program."""

import os
import shutil
import subprocess


def write_outname(
    l_x: int,
    n_ph: int,
    sigma_u: float,
    gam: float,
    omega: float,
    seed_id: int,
    # g_aa: float = None,
    fp: int = 2,
) -> str:
    """Creates a string with the outname for the given parameters.

    Args:
        l_x (int): Chain's size
        g_aa (float): Disorder parameter
        gam (float): Coupling parameter between cavity and chain
        n_ph (int): Maximum number of photons in the cavity
        omega (float): Photon frequency
        fp (int): Precision of the outname float numbers

    Returns:
        str: Outname for the given parameters
    """
    name = ""
    name += f"L{l_x}"
    name += f"Nph{n_ph}"
    name += f"dU{sigma_u:.{fp}f}"
    name += f"gam{gam:.{fp}f}"
    name += f"omega{omega:.{fp}f}"
    name += f"v{seed_id:03}"
    return name.replace(".", "")


def save_readin(
    file: str,
    l_x: int,
    n_ph: int,
    delta_u: float,
    gam: float,
    omega: float,
    ne_points: int,
    n_disorder: int,
    seed: int,
    seed_id: int,
    distkind: int = 2,
    t: float = 1.0,
    tc_s: float = 1.0,
    tc_d: float = 1.0,
    tl_s: float = 1.0,
    tl_d: float = 1.0,
    mu_s: float = 0.0,
    mu_d: float = 0.0,
) -> None:
    """Saves the read.in file for the given parameters."""
    outname = write_outname(
        l_x=l_x,
        n_ph=n_ph,
        sigma_u=delta_u,
        gam=gam,
        omega=omega,
        seed_id=seed_id,
    )
    string_tuple = (
        outname,
        f"{l_x} {n_ph} {seed} {distkind}",
        f"{ne_points} {n_disorder}",
        f"{t} {gam} {delta_u} {omega}",
        f"{tc_s} {tc_d} {tl_s} {tl_d} {mu_s} {mu_d}",
        "",
        "outname",
        "Lx, Nph, seed, distkind",
        "NEpoints, Ndisorder",
        "t, gam, sigmaU, omega",
        "tcS,tcD,tlS,tlD,muS,muD",
        "",
        "distkind=1 --> Gaussian",
        "distkind=/=1 --> uniform",
        "NEpoints defines a grid of (2*NEpoints -2) points",
        "",
    )

    complete_string = "\n".join(string_tuple)
    with open(file=file, mode="w", encoding="utf-8") as f:
        f.write(complete_string)


def submit_job(
    dir: str,
    gam: float,
    sigma_u: float,
    seed: int,
    seed_id: int,
    l_x: int = 100,
    n_ph: int = 5,
    omega: float = 1.0,
    cpus: int = 2,
    ne_points: int = 100,
    n_disorder: int = 1,
) -> None:
    """Submit the job.sh to the slurm."""
    # Create the directory
    outname = write_outname(
        l_x=l_x,
        n_ph=n_ph,
        sigma_u=sigma_u,
        gam=gam,
        omega=omega,
        seed_id=seed_id,
    )
    directory = f"{dir}/data_local/{outname}"
    os.makedirs(directory, exist_ok=True)

    save_readin(
        file=f"{directory}/readin.in",
        l_x=l_x,
        n_ph=n_ph,
        delta_u=sigma_u,
        gam=gam,
        omega=omega,
        ne_points=ne_points,
        n_disorder=n_disorder,
        seed=seed,
        seed_id=seed_id,
    )

    shutil.copy2(
        src=f"{dir}/chainv031new.x",
        dst=directory,
    )

    # os.chdir(directory)
    subprocess.run(
        "./chainv031new.x < readin.in",
        shell=True,
        check=False,
        cwd=directory,
    )


if __name__ == "__main__":
    DIR = os.path.abspath(os.path.dirname(__file__))
    os.chdir(DIR)

    # gammas = [0.04, 0.10, 0.16]
    gammas = [
        # 0.05,
        # 0.10,
        0.15,
        # 0.20,
        # 0.25,
        # 0.30,
    ]
    # sigmas = [
    #     0.1,
    #     0.2,
    #     0.3,
    #     0.4,
    #     # 0.5,
    #     0.6,
    #     0.7,
    #     0.8,
    # ]
    sigmas = [
        0.5,
        # 0.4,
        # 0.6,
        # 0.3,
        # 0.7,
        # 0.2,
        # 0.8,
        # 0.1,
    ]
    seed_0 = 8421
    seed_num = 100
    seeds = list(range(seed_0, seed_0 + seed_num))

    for sigma_u in sigmas:
        for gamma in gammas:
            print(f"sigma = {sigma_u}  gamma = {gamma}")
            for index, seed in enumerate(seeds):
                seed_id = index + 1
                submit_job(
                    dir=DIR,
                    gam=gamma,
                    sigma_u=sigma_u,
                    seed=seed,
                    seed_id=seed_id,
                )
