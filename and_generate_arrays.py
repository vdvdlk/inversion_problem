import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def parse_val(v):
    try:
        return float(v) if ("." in v or "e" in v.lower()) else int(v)
    except ValueError:
        return v


def read_anderson_variables(
    dados_dat_path: Path,
) -> dict[str, float]:
    with open(dados_dat_path, "r") as f:
        texto = f.read()

    texto = texto.replace("\n", " ")
    texto = texto.replace(" Input data  Lx=", "l_x=")
    texto = texto.replace(" Nph=", "\nn_ph=")
    texto = texto.replace(" seed=", "\nseed=")
    texto = texto.replace(
        " Distribution (1 Gaussian, any other retangular)=",
        "\ndistkind=",
    )
    texto = texto.replace(" Energy grid=", "\nne_points=")
    texto = texto.replace(" Number of disorder conf.=", "\nn_disorder=")
    texto = texto.replace(" t=", "\nt=")
    texto = texto.replace(" gam=", "\ngamma=")
    texto = texto.replace(" sigma (dist)=", "\nsigma=")
    texto = texto.replace(" Omega=", "\nomega=")
    texto = texto.replace(" tcS=", "\ntc_s=")
    texto = texto.replace(" tcD=", "\ntc_d=")
    texto = texto.replace(" tlS=", "\ntl_s=")
    texto = texto.replace(" tlD=", "\ntl_d=")
    texto = texto.replace(" muD=", "\nmu_s=", 1)
    texto = texto.replace(" muD=", "\nmu_d=")

    matches = re.findall(r"([\w_]+)=\s*([^\s]+)", texto)

    variables = {k: parse_val(v) for k, v in matches}

    return variables


def get_highest_jobid_dir(job_dir: Path) -> Optional[Path]:
    """Retorna o subdiretório de jobid com o maior número dentro de outname_dir."""
    jobid_dirs = [d for d in job_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not jobid_dirs:
        return None
    return max(jobid_dirs, key=lambda d: int(d.name))


def copy_jobid_dir(job_dir: Path, clean_jobid_dir: bool = False):
    """Copia arquivos do maior jobid para o outname_dir."""
    jobid_dir = get_highest_jobid_dir(job_dir)

    if jobid_dir:
        # Copia todos os arquivos do jobid_dir para o outname_dir
        for file in jobid_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, job_dir)

        # Se quiser apagar o jobid_dir depois de copiar:
        if clean_jobid_dir:
            shutil.rmtree(jobid_dir)


def read_job_dir(
    job_dir: Path,
):
    # ) -> (NDArray[np.float_], dict[str, float]):
    """
    Lê os arquivos de dados de um diretório job e retorna um array numpy
    contendo os dados.
    """
    # Verifica se o diretório existe
    if not job_dir.is_dir():
        raise ValueError(f"Directory {job_dir} does not exist.")

    dados_files = list(job_dir.glob("dados*"))
    if len(dados_files) == 0:
        raise FileNotFoundError(f"Nenhum arquivo 'dados*' encontrado em: {job_dir}")
    elif len(dados_files) > 1:
        raise RuntimeError(
            f"Esperado apenas um arquivo 'dados*' no diretório, mas {len(dados_files)} foram encontrados: {dados_files}"
        )
    variables = read_anderson_variables(dados_files[0])

    bare_data_files = list(job_dir.glob("bareTTlogTT*.dat"))
    if len(bare_data_files) == 0:
        raise FileNotFoundError("Nenhum arquivo 'bareTTlogTT*' encontrado.")
    elif len(bare_data_files) > 1:
        raise RuntimeError(
            f"Esperado apenas um arquivo 'bareTTlogTT*' no diretório, mas {len(bare_data_files)} foram encontrados: {bare_data_files}"
        )

    # bare_energies = np.loadtxt(
    #     fname=bare_data_files[0],
    #     usecols=0,
    # )
    # bare_tt = np.loadtxt(
    #     fname=bare_data_files[0],
    #     usecols=1,
    # )
    dat_array = np.loadtxt(fname=bare_data_files[0], dtype=np.float_)
    bare_energies = dat_array[:, 0]
    bare_tt = dat_array[:, 1]

    return bare_energies, bare_tt, variables


def bare_data_to_array(
    bare_data: NDArray[np.float_],
    n_disorder: int,
) -> NDArray[np.float_]:
    """
    Converte os dados de bare_data para um array numpy. O primeiro eixo do array
    resultante representa os pontos de energia, enquanto o segundo eixo representa
    as realizações de desordem.
    """
    # Divide o array em partes iguais de tamanho n_disorder
    splitted_arrays = np.split(bare_data, bare_data.size // n_disorder)

    # Empilha os arrays resultantes em um novo array 2D
    return np.stack(splitted_arrays, axis=0)


def compressed_to_ca_arrays(zip_path: Path, copy_from_jobid: bool = False):
    """Converts a zip file to an array"""
    # Cria um diretório temporário
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)
        print(rf"Diretório temporário criado: {tmp_dir}")
        print(rf"Extraindo arquivo: {zip_path}")

        # Extrai o conteúdo do arquivo zip para o diretório temporário
        shutil.unpack_archive(
            filename=zip_path,
            extract_dir=tmp_dir,
            # format="zip",
        )

        extracted_paths = list(tmp_dir.iterdir())
        if len(extracted_paths) == 0:
            raise FileNotFoundError("Nenhum diretório dentro do arquivo comprimido.")
        elif len(extracted_paths) > 1:
            raise RuntimeError(
                f"Esperado apenas um diretório extraído, mas {len(extracted_paths)} foram encontrados: {extracted_paths}"
            )
        extracted_dir = extracted_paths[0]
        print(rf"Arquivo zip extraído: {extracted_dir}")

        tt_arrays_1 = []
        energy_arrays_1 = []
        sigmas_1 = []
        gammas_1 = []
        desc = f"Lendo {zip_path}"
        for sigma_dir in tqdm(sorted(extracted_dir.iterdir()), desc=desc):
            if not sigma_dir.is_dir():
                continue
            # print(sigma_dir)

            tt_arrays_2 = []
            energy_arrays_2 = []
            sigmas_2 = []
            gammas_2 = []
            for gam_sig_dir in sorted(sigma_dir.iterdir()):
                if not gam_sig_dir.is_dir():
                    continue

                if copy_from_jobid:
                    copy_jobid_dir(job_dir=gam_sig_dir)

                bare_energies, bare_tt, variables = read_job_dir(
                    job_dir=gam_sig_dir,
                )
                # print(f"{bare_energies.shape=}")
                # print(f"{bare_tt.shape=}")
                # print(f"{variables=}")
                n_disorder = variables["n_disorder"]

                sigmas_2.append(variables["sigma"])
                gammas_2.append(variables["gamma"])

                energy_arrays_2.append(
                    bare_data_to_array(
                        bare_data=bare_energies,
                        n_disorder=n_disorder,
                    )
                )
                # print(energy_arrays_2)
                tt_arrays_2.append(
                    bare_data_to_array(
                        bare_data=bare_tt,
                        # ne_points=ne_points,
                        n_disorder=n_disorder,
                    )
                )
            energy_arrays_1.append(np.stack(energy_arrays_2))
            tt_arrays_1.append(np.stack(tt_arrays_2))
            sigmas_1.append(sigmas_2)
            gammas_1.append(gammas_2)

    energies = np.stack(energy_arrays_1)
    tts = np.stack(tt_arrays_1)
    sigmas = np.stack(sigmas_1)
    gammas = np.stack(gammas_1)

    return energies, tts, sigmas, gammas


def save_ca_arrays(
    zip_path: Path,
    array_path: Path,
    copy_from_jobid: bool = False,
):
    energies, tts, sigmas, gammas = compressed_to_ca_arrays(
        zip_path=zip_path,
        copy_from_jobid=copy_from_jobid,
    )
    print("Salvando array")
    np.savez_compressed(
        file=array_path,
        energies=energies[0, 0, :, 0],
        tts=tts,
        sigmas=sigmas[:, 0],
        gammas=gammas[0, :],
    )
    print(f"Array salvo em {array_path}\n")


def organize_versions(data_dir: Path):
    """Move diretórios *vNNN para dentro de um diretório de mesmo prefixo."""
    desc = "Organizando versões"
    dir_iterable = sorted(data_dir.iterdir())

    for dirpath in tqdm(iterable=dir_iterable, desc=desc):
        if not dirpath.is_dir():
            continue

        name = dirpath.name
        if "v" not in name:
            continue  # Ignora diretórios que não seguem o padrão

        # Extrai o prefixo (ex: 'dir1' de 'dir1_v001')
        prefix = name.split("v")[0]
        target_dir = data_dir / prefix

        # Cria o diretório do prefixo se ainda não existir
        target_dir.mkdir(exist_ok=True)

        # Move o diretório atual para dentro do diretório prefixo
        shutil.move(str(dirpath), str(target_dir / name))


def compressed_to_input_arrays(
    tar_path: Path,
):
    """Converts a zip file to an array"""
    # Cria um diretório temporário
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)
        print(rf"Diretório temporário criado: {tmp_dir}")
        print(rf"Extraindo arquivo: {tar_path}")

        # Extrai o conteúdo do arquivo tar para o diretório temporário
        shutil.unpack_archive(
            filename=tar_path,
            extract_dir=tmp_dir,
        )

        extracted_paths = list(tmp_dir.iterdir())
        if len(extracted_paths) == 0:
            raise FileNotFoundError("Nenhum diretório dentro do .tar.")
        elif len(extracted_paths) > 1:
            raise RuntimeError(
                f"Esperado apenas um diretório extraído, mas {len(extracted_paths)} foram encontrados: {extracted_paths}"
            )
        extracted_dir = extracted_paths[0]
        print(rf"Arquivo tar extraído: {extracted_dir}")

        organize_versions(extracted_dir)

        energy_arrays_1 = []
        tt_arrays_1 = []
        sigmas_1 = []
        gammas_1 = []
        dir_iterable = sorted(extracted_dir.iterdir())
        for outname_dir in tqdm(dir_iterable, desc="Lendo arquivos de dados"):
            if not outname_dir.is_dir():
                continue

            energy_arrays_2 = []
            tt_arrays_2 = []
            for v_dir in sorted(outname_dir.iterdir()):
                if not v_dir.is_dir():
                    continue
                copy_jobid_dir(v_dir)

                bare_energies, bare_tt, variables = read_job_dir(
                    job_dir=v_dir,
                )

                energy_arrays_2.append(bare_energies)
                tt_arrays_2.append(bare_tt)
            sigmas_1.append(variables["sigma"])
            gammas_1.append(variables["gamma"])
            energy_arrays_1.append(np.stack(energy_arrays_2))
            tt_arrays_1.append(np.stack(tt_arrays_2))

    energies = np.stack(energy_arrays_1)
    tts = np.stack(tt_arrays_1)
    sigmas = np.stack(sigmas_1)
    gammas = np.stack(gammas_1)

    return energies, tts, sigmas, gammas


def save_input_arrays(
    tar_path: Path,
    array_path: Path,
):
    energies, tts, sigmas, gammas = compressed_to_input_arrays(tar_path=tar_path)
    print("Salvando array")
    np.savez_compressed(
        file=array_path,
        energies=energies[0, 0, :],
        tts=tts,
        sigmas=sigmas,
        gammas=gammas,
    )
    print(f"Array salvo em {array_path}\n")


if __name__ == "__main__":
    DIR = Path(__file__).parent

    AND_COMP_DIR = DIR / "compressed_data" / "anderson"
    AND_ARRAY_DIR = DIR / "arrays" / "anderson"
    AND_ARRAY_DIR.mkdir(exist_ok=True, parents=True)

    # L = 100
    save_ca_arrays(
        zip_path=AND_COMP_DIR / "dados_IP.zip",
        array_path=AND_ARRAY_DIR / "dados_IP.npz",
    )
    save_input_arrays(
        tar_path=AND_COMP_DIR / "data_anderson_L100.tar.xz",
        array_path=AND_ARRAY_DIR / "data_anderson_L100.npz",
    )

    # L = 200
    save_input_arrays(
        tar_path=AND_COMP_DIR / "data_anderson_L200.tar.xz",
        array_path=AND_ARRAY_DIR / "data_anderson_L200.npz",
    )
    save_ca_arrays(
        zip_path=AND_COMP_DIR / "data_anderson_ca_L200_gamma015.tar.xz",
        array_path=AND_ARRAY_DIR / "data_anderson_ca_L200_gamma015.npz",
        copy_from_jobid=True,
    )
    save_ca_arrays(
        zip_path=AND_COMP_DIR / "data_anderson_ca_L200_W050.tar.xz",
        array_path=AND_ARRAY_DIR / "data_anderson_ca_L200_W050.npz",
        copy_from_jobid=True,
    )
