pipenv install
pipenv run pre-commit autoupdate
pipenv run pre-commit install -f --install-hooks
source setup.sh
