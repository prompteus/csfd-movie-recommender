# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: 'Python 3.9.7 (''env'': venv)'
#     language: python
#     name: python3
# ---

# %%

import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import bs4
import pandas as pd
import requests
from pandarallel import pandarallel
from tqdm import tqdm

from recommend.utils import PROJ_ROOT

tqdm.pandas()
pandarallel.initialize(progress_bar=True)

# %%
ratings = pd.read_pickle("{PROJ_ROOT}/data/ratings.pkl")

# %%
movie_ids = pd.Series(ratings.movie_id.dropna().unique())

# %%
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
}


def download_movie_info(movie_id: str) -> requests.Response:
    wait_secs = random.randint(1, 5) / 10
    time.sleep(wait_secs)
    response = requests.get(f"https://www.csfd.cz/film/{movie_id}", headers=headers)
    if response.status_code != 200:
        print(response.status_code)
    return response


# %%
movie_htmls_path = f"{PROJ_ROOT}/data/movies_htmls.pkl"
if os.path.exists(movie_htmls_path):
    htmls = pd.read_pickle(movie_htmls_path)
else:
    responses = movie_ids.parallel_apply(download_movie_info)
    htmls = responses.apply(lambda response: response.text)
    htmls.to_pickle(movie_htmls_path)

# %%
htmls


# %%
def extract_title(doc: bs4.BeautifulSoup) -> str:
    return doc.find("h1").text.strip()


def extract_avg_rating(doc: bs4.BeautifulSoup) -> str:
    return doc.select_one(".rating-average").text.strip()


def extract_kind(doc: bs4.BeautifulSoup) -> str:
    return doc.select_one("span.type").text.strip().strip("()").strip()


def extract_genres(doc: bs4.BeautifulSoup) -> List[str]:
    return [genre.strip() for genre in doc.select_one(".genres").text.strip().split("/")]


def extract_countries(doc: bs4.BeautifulSoup) -> List[str]:
    return [
        country.strip()
        for country in list(doc.select_one(".origin").children)[0]
        .strip()
        .strip(",")
        .strip()
        .split("/")
    ]


def extract_year(doc: bs4.BeautifulSoup) -> str:
    return list(doc.find(attrs={"class": "origin"}).children)[1].text.strip().strip(",").strip()


def extract_length(doc: bs4.BeautifulSoup) -> str:
    return list(doc.find(attrs={"class": "origin"}).children)[2].text.strip().strip(",").strip()


def extract_poster(doc: bs4.BeautifulSoup) -> str:
    return doc.select_one(".film-posters img")["srcset"].split(",")[-1].strip().split(" ")[0]


def extract_description(doc: bs4.BeautifulSoup) -> str:
    return (
        doc.select_one(".plot-full p")
        .text.replace(doc.select_one(".plot-full em").text, "")
        .strip()
    )


def extract_foreign_titles(doc: bs4.BeautifulSoup) -> Dict[str, str]:
    return {
        item.select_one("img")["title"]: "".join(item.find_all(text=True, recursive=False)).strip()
        for item in doc.select(".film-names li")
    }


def extract_creators(doc: bs4.BeautifulSoup) -> Dict[str, List[Tuple[str, str]]]:
    return {
        item.select_one("h4")
        .text.strip()
        .strip(":"): [(a["href"], a.text) for a in item.select("a:not(.more)")]
        for item in doc.select(".creators div")
    }


def try_or_none(fun: Callable) -> Callable:
    def f(x: Any) -> Optional[Any]:
        try:
            return fun(x)
        except Exception:
            return None

    return f


extraction = {
    "title": extract_title,
    "description": extract_description,
    "kind": extract_kind,
    "genres": extract_genres,
    "countries": extract_countries,
    "year": extract_year,
    "length": extract_length,
    "year": extract_year,
    "poster": extract_poster,
    "foreign_titles": extract_foreign_titles,
    "creators": extract_creators,
}

# %%
docs = htmls.progress_apply(lambda html: bs4.BeautifulSoup(html, "lxml"))

# %%
docs

# %%
movies = pd.DataFrame(
    {name: docs.progress_apply(try_or_none(fun)) for name, fun in extraction.items()}
)
movies

# %%
movies.set_index(movie_ids, drop=True, inplace=True)
movies.index.rename("movie_id", inplace=True)
movies.kind = movies.kind.apply(lambda val: "series" if val == "seriál" else "movie")
movies.foreign_titles = movies.foreign_titles.apply(lambda titles: {} if titles is None else titles)
movies.creators = movies.creators.apply(lambda creators: {} if creators is None else creators)
movies.description = movies.description.fillna("")
movies.poster = movies.poster.fillna("")
movies

# %%
movies.to_pickle(f"{PROJ_ROOT}/data/movies.pkl")
