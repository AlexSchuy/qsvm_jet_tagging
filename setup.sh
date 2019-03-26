PROJECTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export PYTHONPATH="${PYTHONPATH}:${PROJECTPATH}/qsvm_jet_tagging"
pipenv shell
