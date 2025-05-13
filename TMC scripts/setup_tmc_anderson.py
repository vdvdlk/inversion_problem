"""Script em Python para automatizar o envio de jobs para o cluster TMC."""

import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import List


def compile_f90(
    source_dir: Path,
    program_name: str,
    # compiler: str = "ifort",
) -> None:
    """Compila o programa em Fortran.
    Args:
        source_dir (str): Diretório onde está o código fonte.
        program_name (str): Nome do arquivo de código fonte.
    """
    f90_path = source_dir / program_name
    if not f90_path.is_file():
        raise FileNotFoundError(f"Arquivo {f90_path} não encontrado.")

    compile_command = [
        "ifort",
        # "-diag-disable=10448",
        "-qmkl",
        f"{program_name}.f90",
        "-o",
        f"{program_name}.x",
    ]
    subprocess.run(
        args=compile_command,
        check=True,
        cwd=source_dir,
    )

    # a_out_path = source_dir / "a.out"
    # if not a_out_path.is_file():
    #     raise FileNotFoundError(
    #         f"Arquivo {a_out_path} não encontrado após a compilação."
    #     )


def generate_outname(
    l_x: int,
    n_ph: int,
    sigma_u: float,
    gam: float,
    omega: float,
    seed_id: int,
    # g_aa: float = None,
    fp: int = 2,
) -> str:
    """Cria uma string com o outname para os parâmetros dados.

    Args:
        l_x (int): Tamanho do sistema
        g_aa (float): Parâmetro de desordem
        gam (float): Parâmetro de acoplamento entre a cavidade e o sistema
        n_ph (int): Número máximo de fótons na cavidade
        omega (float): Frequência do fóton
        fp (int): Precisão da representação decimal no outname

    Returns:
        str: outname para os parâmetros dados
    """
    outname = f"L{l_x}Nph{n_ph}dU{sigma_u:.{fp}f}gam{gam:.{fp}f}omega{omega:.{fp}f}"

    if seed_id is not None:
        outname += f"v{seed_id:03}"

    outname = outname.replace(".", "")
    return outname


def generate_readin(
    job_dir: Path,
    outname: str,
    l_x: int,
    n_ph: int,
    delta_u: float,
    gam: float,
    omega: float,
    ne_points: int,
    n_disorder: int,
    seed: int,
    # seed_id: int,
    distkind: int = 2,
    t: float = 1.0,
    tc_s: float = 1.0,
    tc_d: float = 1.0,
    tl_s: float = 1.0,
    tl_d: float = 1.0,
    mu_s: float = 0.0,
    mu_d: float = 0.0,
    # fp: int = 2,
) -> None:
    """Gera o arquivo read.in dentro do diretório do job e retorna o outname.

    Args:
        job_dir (Path): Diretório onde o read.in será salvo.
        fp (int): Precisão dos números de ponto flutuante no outname.
        [restante: parâmetros físicos do sistema]
    """
    file_content = textwrap.dedent(f"""\
        {outname}
        {l_x} {n_ph} {seed} {distkind}
        {ne_points} {n_disorder}
        {t} {gam} {delta_u} {omega}
        {tc_s} {tc_d} {tl_s} {tl_d} {mu_s} {mu_d}

        outname
        Lx, Nph, seed, distkind
        NEpoints, Ndisorder
        t, gam, sigmaU, omega
        tcS,tcD,tlS,tlD,muS,muD

        distkind=1 --> Gaussian
        distkind=/=1 --> Uniform
        NEpoints defines a grid of (2*NEpoints - 2) points
    """)

    file_path = job_dir / "read.in"
    # file_path.write_text(
    #     data="\n".join(lines),
    #     encoding="utf-8",
    # )
    file_path.write_text(
        data=file_content,
        encoding="utf-8",
    )


def generate_jobfile(
    job_dir: Path,
    jobname: str,
    cpus: int,
) -> None:
    """Gera o arquivo job.sh para submissão via SLURM, usando template string.

    Args:
        job_dir (Path): Diretório onde o job.sh será salvo.
        jobname (str): Nome do job no SLURM.
        cpus (int, optional): Número de CPUs por tarefa. Padrão é 1.
    """
    file_content = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -N 1                       # Number of nodes
        #SBATCH -n 1                       # Number of tasks
        #SBATCH --cpus-per-task={cpus}     # CPUs per task
        #SBATCH -J {jobname[:8]}           # Job name (max 8 characters)

        ###################################
        ########## Bookkeeping ############
        ###################################

        EXECFILE="a.out"
        PARAMLIST="read.in"
        IFILELIST="$EXECFILE $PARAMLIST"

        echo "Executing at  : $SLURM_JOB_NODELIST"
        echo -n "Date          : "
        date
        echo "Job ID        : $SLURM_JOBID"
        echo "Job Name      : $SLURM_JOB_NAME"
        echo "User Name     : $SLURM_JOB_USER"
        echo "Home Directory: $SLURM_SUBMIT_DIR"
        echo "CPUs per task : $SLURM_CPUS_PER_TASK"

        ###################################
        ######## Preparing to run #########
        ###################################

        # module load intel
        source /opt/intel/oneapi/setvars.sh
        echo "Creating a directory to run the program"
        mkdir $SLURM_JOBID

        echo "Moving input files to run directory"
        for i in $IFILELIST
        do
            scp $i $SLURM_JOBID
        done
        chmod +x $EXECFILE

        ###################################
        ########## Running program ########
        ###################################

        cd $SLURM_JOBID
        ulimit -s unlimited
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

        echo "Running"
        time srun ./$EXECFILE < $PARAMLIST

        echo "Finished at : $SLURM_JOB_NODELIST"
        echo -n "Date         : "
        date
    """)

    jobfile_path = job_dir / "job.sh"
    jobfile_path.write_text(data=file_content, encoding="utf-8")


def submit_job(
    data_dir: Path,
    source_dir: Path,
    jobname: str,
    l_x: int,
    n_ph: int,
    delta_u: float,
    gam: float,
    omega: float,
    cpus: int,
    seed: int,
    seed_id: int = None,
    program_name: str = "chainv031new",
    ne_points: int = 100,
    n_disorder: int = 1,
    distkind: int = 2,
) -> None:
    """Prepara o diretório, gera os arquivos e submete um job para o SLURM."""

    # 1. Gera o outname único para este job
    outname = generate_outname(
        l_x=l_x,
        n_ph=n_ph,
        sigma_u=delta_u,
        gam=gam,
        omega=omega,
        seed_id=seed_id,
    )

    # 2. Cria o diretório do job
    job_dir = data_dir / outname
    job_dir.mkdir(parents=True, exist_ok=True)

    # 3. Gera o arquivo read.in e confirma o outname
    generate_readin(
        job_dir=job_dir,
        outname=outname,
        l_x=l_x,
        n_ph=n_ph,
        delta_u=delta_u,
        gam=gam,
        omega=omega,
        ne_points=ne_points,
        n_disorder=n_disorder,
        seed=seed,
        distkind=distkind,
    )

    # 4. Gera o arquivo job.sh
    generate_jobfile(
        job_dir=job_dir,
        jobname=jobname,
        cpus=cpus,
    )

    # 5. Compila o programa
    # source_dir = Path(__file__).resolve().parent
    # compile_f90(source_dir=source_dir, program_name=program_name)

    # 6. Copia o a.out para o diretório do job
    a_out_path = source_dir / f"{program_name}.x"
    dest_path = job_dir / "a.out"
    shutil.copy2(src=a_out_path, dst=dest_path)

    # 7. Submete o job
    subprocess.run(
        ["sbatch", str(job_dir / "job.sh")],
        check=True,
        cwd=job_dir,
    )


def submit_ca_jobs(
    data_dir: Path,
    source_dir: Path,
    program_name: str,
    cpus: int,
    l_x: int,
    delta_u_list: List[float],
    gam_list: List[float],
    omega: float = 1.0,
    seed: int = 1895,
    n_ph: int = 5,
    ne_points: int = 100,
    n_disorder: int = 1000,
    distkind: int = 2,
) -> None:
    compile_f90(
        source_dir=source_dir,
        program_name=program_name,
    )
    for delta_u in delta_u_list:
        for gam in gam_list:
            jobname = f"W{delta_u:.1f}g{gam:.2f}".replace(".", "")

            submit_job(
                data_dir=data_dir,
                source_dir=source_dir,
                jobname=jobname,
                l_x=l_x,
                n_ph=n_ph,
                delta_u=delta_u,
                gam=gam,
                omega=omega,
                seed=seed,
                program_name=program_name,
                cpus=cpus,
                ne_points=ne_points,
                n_disorder=n_disorder,
                distkind=distkind,
            )


if __name__ == "__main__":
    SETUPS_DIR = Path(__file__).resolve().parent
    DIR = SETUPS_DIR.parent
    SOURCE_DIR = DIR / "Sources"
    PROGRAM_NAME = "chainv031new"

    JULIAN_SIGMAS = [
        # 0.1,
        # 0.2,
        0.3,
        # 0.4,
        # 0.5,
        # 0.6,
        # 0.7,
        # 0.8,
    ]
    JULIAN_GAMMAS = [
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.14,
        0.15,
        0.16,
        0.20,
        0.24,
        0.26,
        0.28,
        0.30,
    ]

    # sigma_gamma_pairs = [
    #     # (0.2, 0.08),
    #     # (0.6, 0.08),
    #     (0.2, 0.22),
    #     # (0.6, 0.22),
    #     # (0.5, 0.15),
    # ]

    # seed_0 = 8421  # 8421 / CA: 1829
    # seed_num = 200
    # seeds = list(range(seed_0, seed_0 + seed_num))
    # seed_ids = list(range(1, seed_num + 1))

    # cluster_num = 50
    # step = 0

    # for sigma_u, gamma in sigma_gamma_pairs:
    #     # for i in range(cluster_num):
    #     #     seed = seeds[step * cluster_num + i]
    #     #     seed_id = seed_ids[step * cluster_num + i]
    #     #     # print(seed, seed_id)
    #     #     submit_job(
    #     #         data_dir=DATA_DIR,
    #     #         jobname=f"AnS{step}S{seed_id:03}",
    #     #         l_x=200,
    #     #         n_ph=5,
    #     #         gam=gamma,
    #     #         omega=1.0,
    #     #         delta_u=sigma_u,
    #     #         seed=seed,
    #     #         seed_id=seed_id,
    #     #         n_disorder=1,
    #     #         cpus=4,
    #     #     )

    #     for index, seed in enumerate(seeds):
    #         seed_id = index + 1
    #         submit_job(
    #             data_dir=DATA_DIR,
    #             jobname=f"AndS{seed_id:03}",
    #             l_x=200,
    #             n_ph=5,
    #             gam=gamma,
    #             omega=1.0,
    #             delta_u=sigma_u,
    #             seed=seed,
    #             seed_id=seed_id,
    #             n_disorder=1,
    #             cpus=4,
    #         )

    submit_ca_jobs(
        data_dir=DIR / "data_anderson_ca_L200_2",
        source_dir=SOURCE_DIR,
        program_name=PROGRAM_NAME,
        cpus=8,
        l_x=200,
        delta_u_list=[0.5],
        gam_list=JULIAN_GAMMAS[7:],
    )
    # submit_ca_jobs(
    #     data_dir=DIR / "data_anderson_ca_L200_2",
    #     source_dir=SOURCE_DIR,
    #     program_name=PROGRAM_NAME,
    #     cpus=8,
    #     l_x=200,
    #     delta_u_list=[0.5],
    #     gam_list=JULIAN_GAMMAS[7:],
    # )
