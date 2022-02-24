# Recommender system

## Environment

To setup the environment, use Python 3.9.7 to create virtual env, activate it,
install the requirements and this package in editable mode:

```
git clone git@github.com:markcheeky/csfd-movie-recommender.git 
cd csfd-movie-recommender
python3 -m venv env
source env/bin/activate
make install
```

The code should be formatted with black + isort (and ideally have type hints),
so make sure to set up your IDE accordingly. If you use VS Code, it should
pick up the settings automatically and format on save. If you want to quickly
prototype with jupyter notebooks use `src/recommend/notebooks/` directory.
All ipynb files are gitignored, but you can run `make jupytext file=...`
to convert notebook into a plain .py file which is easy to diff.

To check the code (format + style + tests), run `make check`.
