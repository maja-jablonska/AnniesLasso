#!/usr/bin/env bash
# Central qsub wrapper for AnniesLasso PBS jobs on Gadi/NCI.
#
# Mirrors bulge-ages-and-orbits/pbs/qsub.sh: the directives every job shares
# (mail recipient, mail events, joined output) live HERE instead of being
# repeated -- and drifting -- across every job script. Job-specific directives
# (queue, ncpus, mem, walltime, storage) stay in the individual scripts.
#
# Use it exactly like qsub:
#
#   bash scripts/qsub.sh -v OUT_ROOT=$HOME/scr_mk27/benchmark_v8 scripts/compare_state_classifiers.pbs
#
# The project account is NOT forced: the wrapper reads the '#PBS -P' line of
# the script being submitted and passes it through, so a job keeps whatever
# allocation it declares. Jobs in these repos are split between mk27 and dg97,
# and a wrapper that hard-coded one would silently re-bill half of them.
#
# The override variable is QSUB_PROJECT, deliberately NOT $PROJECT: Gadi
# exports $PROJECT as your default project (y89), so a wrapper reading it
# would override every script's own -P with y89 and fail against any queue
# y89 cannot use -- while your edit to the '#PBS -P' line appeared to do
# nothing. You can also just pass -P after the wrapper (qsub options beat
# #PBS directives, and a later option beats an earlier one).
#
#   QSUB_PROJECT=dg97 bash scripts/qsub.sh -v OUT_ROOT=... scripts/foo.pbs
#   bash scripts/qsub.sh -P dg97 -v OUT_ROOT=... scripts/foo.pbs
#   bash scripts/qsub.sh -m n -v ... scripts/foo.pbs        # this run: no mail
#
# The wrapped defaults:
#   -P <the script's own, else mk27>   project account (QSUB_PROJECT overrides)
#   -M maja.jablonska@anu.edu.au       mail recipient
#   -m bae                             mail on begin / abort / end
#   -j oe                              merge stderr into stdout

set -uo pipefail

# The job script is the last argument; the project it declares wins unless
# QSUB_PROJECT says otherwise. Never read $PROJECT -- Gadi sets it.
script="${!#}"
project="${QSUB_PROJECT:-}"
if [ -z "${project}" ] && [ -f "${script}" ]; then
    project=$(awk '/^#PBS -P /{print $3; exit}' "${script}")
fi
project="${project:-mk27}"

exec qsub \
    -P "${project}" \
    -M maja.jablonska@anu.edu.au \
    -m bae \
    -j oe \
    "$@"
