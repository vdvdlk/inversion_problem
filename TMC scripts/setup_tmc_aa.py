"""Script em Python para automatizar o envio de jobs para o cluster TMC."""

import shutil
import subprocess
import textwrap
from math import sqrt
from pathlib import Path


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

    a_out_path = source_dir / "a.out"

    compile_command = [
        "ifort",
        "-diag-disable=10448",
        "-qmkl",
        str(f90_path),
        "-o",
        str(a_out_path),
    ]
    subprocess.run(
        args=compile_command,
        check=True,
        # cwd=source_dir,
    )

    if not a_out_path.is_file():
        raise FileNotFoundError(
            f"Arquivo {a_out_path} não encontrado após a compilação."
        )


def prepare_directory(data_dir: Path, outname: str) -> Path:
    """Cria o diretório onde os arquivos do job serão armazenados.

    Args:
        data_dir (Path): Caminho do diretório de dados.
        outname (str): Nome específico do job.

    Returns:
        Path: Caminho completo para o diretório do job criado.
    """
    job_dir = data_dir / outname
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def generate_outname(
    l_x: int,
    n_ph: int,
    g_aa: float,
    gamma: float,
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
    outname = (
        f"L{l_x}Nph{n_ph}g{g_aa:.{fp}f}gam{gamma:.{fp}f}omega{omega:.{fp}f}v{seed_id:03}"
    ).replace(".", "")
    return outname


def generate_readin(
    job_dir: Path,
    outname: str,
    l_x: int,
    n_ph: int,
    g_aa: float,
    gamma: float,
    omega: float,
    ne_points: int,
    n_disorder: int,
    seed: int,
    beta: float = (sqrt(5) - 1) / 2,
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
        {l_x} {n_ph} {ne_points} {n_disorder}
        {seed}
        {g_aa} {beta}
        {t} {gamma} {omega}
        {tc_s} {tc_d} {tl_s} {tl_d} {mu_s} {mu_d}

        outname
        Lx, Nph, NEpoints, Ndisorder
        seed
        gAA, beta
        t, gam, omega
        tcS,tcD,tlS,tlD,muS,muD

        NEpoints defines a grid of (2*NEpoints -2) points
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
    cpus: int = 1,
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

        echo "Creating a directory to run the program"
        mkdir $SLURM_JOBID

        echo "Moving input files to run directory"
        for i in $IFILELIST
        do
            mv $i $SLURM_JOBID
        done

        ###################################
        ########## Running program ########
        ###################################

        cd $SLURM_JOBID
        ulimit -s unlimited
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

        echo "Running"
        chmod +x $EXECFILE
        time srun ./$EXECFILE < $PARAMLIST

        echo "Finished at : $SLURM_JOB_NODELIST"
        echo -n "Date         : "
        date
    """)

    jobfile_path = job_dir / "job.sh"
    jobfile_path.write_text(data=file_content, encoding="utf-8")


def submit_job(
    data_dir: Path,
    jobname: str,
    l_x: int,
    n_ph: int,
    g_aa: float,
    gamma: float,
    omega: float,
    seed: int,
    seed_id: int,
    program_name: str = "chainAAv04.f90",
    cpus: int = 1,
    ne_points: int = 100,
    n_disorder: int = 1,
) -> None:
    """Prepara o diretório, gera os arquivos e submete um job para o SLURM."""

    # 1. Gera o outname único para este job
    outname = generate_outname(
        l_x=l_x,
        n_ph=n_ph,
        g_aa=g_aa,
        gamma=gamma,
        omega=omega,
        seed_id=seed_id,
    )

    # 2. Cria o diretório do job
    job_dir = prepare_directory(data_dir=data_dir, outname=outname)

    # 3. Gera o arquivo read.in e confirma o outname
    generate_readin(
        job_dir=job_dir,
        outname=outname,
        l_x=l_x,
        n_ph=n_ph,
        g_aa=g_aa,
        gamma=gamma,
        omega=omega,
        ne_points=ne_points,
        n_disorder=n_disorder,
        seed=seed,
    )

    # 4. Gera o arquivo job.sh
    generate_jobfile(
        job_dir=job_dir,
        jobname=jobname,
        cpus=cpus,
    )

    # 5. Compila o programa
    source_dir = Path(__file__).resolve().parent
    compile_f90(source_dir=source_dir, program_name=program_name)

    # 6. Copia o a.out para o diretório do job
    a_out_path = source_dir / "a.out"
    dest_path = job_dir / "a.out"
    shutil.copy2(src=a_out_path, dst=dest_path)

    # 7. Submete o job
    subprocess.run(
        ["sbatch", str(job_dir / "job.sh")],
        check=True,
        cwd=job_dir,
    )


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent
    DATA_DIR = DIR / "data_aah"

    g_gamma_pairs = [
        (1.2, 0.10),
        # (1.8, 0.10),
        # (2.4, 0.10),
    ]

    seed_0 = 8421  # 8421 / CA: 1829
    seed_num = 100
    seeds = list(range(seed_0, seed_0 + seed_num))
    seed_ids = list(range(1, seed_num + 1))

    cluster_num = 50
    step = 0

    for g_aa, gamma in g_gamma_pairs:
        # for i in range(cluster_num):
        #     seed = seeds[step * cluster_num + i]
        #     seed_id = seed_ids[step * cluster_num + i]
        #     # print(seed, seed_id)
        #     submit_job(
        #         data_dir=DATA_DIR,
        #         jobname=f"AnS{step}S{seed_id:03}",
        #         l_x=200,
        #         n_ph=5,
        #         gam=gamma,
        #         omega=1.0,
        #         delta_u=sigma_u,
        #         seed=seed,
        #         seed_id=seed_id,
        #         n_disorder=1,
        #         cpus=4,
        #     )

        for index, seed in enumerate(seeds):
            seed_id = index + 1
            submit_job(
                data_dir=DATA_DIR,
                jobname=f"AAHS{seed_id:03}",
                l_x=200,
                n_ph=5,
                gamma=gamma,
                omega=1.0,
                g_aa=g_aa,
                seed=seed,
                seed_id=seed_id,
                n_disorder=1,
                cpus=4,
            )

    # teste_dir = DIR / "teste"
    # submit_job(
    #     data_dir=teste_dir,
    #     jobname="Directory_2",
    #     l_x=100,
    #     n_ph=0,
    #     g_aa=1.0,
    #     gamma=0.0,
    #     omega=0.0,
    #     seed=8421,
    #     seed_id=1,
    #     cpus=1,
    #     n_disorder=1,
    # )
