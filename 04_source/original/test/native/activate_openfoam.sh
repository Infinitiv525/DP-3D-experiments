#!/bin/bash

# This file is sourced by native OpenFOAM Slurm jobs on PERUN.
# It intentionally contains no machine-local user settings.

PROJECT_ROOT="/mnt/project/perun2601449/dp_llm_cfd"
OPENFOAM_ENV="${PROJECT_ROOT}/envs/openfoam-2412"

if [[ ! -x "${OPENFOAM_ENV}/bin/blockMesh" ]]; then
    echo "OpenFOAM environment is missing: ${OPENFOAM_ENV}" >&2
    return 1 2>/dev/null || exit 1
fi

export CONDA_PREFIX="${OPENFOAM_ENV}"
export PATH="${OPENFOAM_ENV}/bin:${PATH}"
export FOAM_MPI_LIBBIN="${OPENFOAM_ENV}/lib/sys-mpich"
export LD_LIBRARY_PATH="${FOAM_MPI_LIBBIN}:${OPENFOAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# The conda-forge package keeps the OpenFOAM-specific variables here.
source "${OPENFOAM_ENV}/etc/conda/activate.d/openfoam_activate.sh"
