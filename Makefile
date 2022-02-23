PYTHON_SOURCES = src tests setup.py
PACKAGE_NAME = recommend

check: black-check isort-check flake8 mypy pytest

fmt: isort black

black:
	black $(PYTHON_SOURCES)

black-check:
	black --check --diff $(PYTHON_SOURCES)

flake8:
	flake8 $(PYTHON_SOURCES)

isort:
	isort $(PYTHON_SOURCES)

isort-check:
	isort --check --diff $(PYTHON_SOURCES)

mypy:
	mypy $(PYTHON_SOURCES)

pytest:
	pytest -v --color=yes --durations=20 --doctest-modules --cov "$(PACKAGE_NAME)" --pyargs "$(PACKAGE_NAME)" tests
