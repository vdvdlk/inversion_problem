#!/bin/bash

# Compilador
#FC="ifort"
FC="ifx"

# Opções de compilação
#FFLAGS="-diag-disable=10448 -O2 -warn all -qmkl"
#FFLAGS="-diag-disable=10448 -O2 -qmkl -qopt-report=5"
FFLAGS="-O2 -qmkl -qopt-report=3"

source /opt/intel/oneapi/setvars.sh

# Loop sobre todos os arquivos .f90 no diretório atual
for src in *.f90; do
    # Extrai o nome base sem extensão
    base="${src%.f90}"
    # Define o nome do executável
    exe="${base}.x"

    echo "Compilando $src -> $exe"
    $FC $FFLAGS "$src" -o "$exe"

    # Verifica se houve erro
    if [ $? -eq 0 ]; then
        echo "✓ Sucesso: $exe criado."
    else
        echo "✗ Erro ao compilar $src."
    fi
done
