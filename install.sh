pipenv install --selective-upgrade numpy
pipenv install --selective-upgrade numpythia
pipenv run pre-commit autoupdate
pipenv run pre-commit install -f --install-hooks
source setup.sh
