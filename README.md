# Recommender system

## Environment

To setup the environment, use Python 3.9.7 to create virtual env, activate it,
install the requirements and this package in editable mode:

```
git clone ...
cd pv254-recommenders
python3 -m venv env
source env/bin/activate
pip install -r requirements-lock.txt
pip install -e .
pre-commit install
```

The code should be formatted with black + isort (and ideally have type hints),
so make sure to set up your IDE accordingly. If you use VS Code, it should
pick up the settings automatically and format on save. If you want to quickly
prototype without bothering with writing quality code, use `src/recommend/notebooks/`
directory.

To check the code (format + style + tests), run `make check`.
