import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset

from recommend.cf_model import Model
from recommend.cf_trainer import RatingDataset, Trainer


class CFRecommender(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def predict_ratings(self, movies: List[str], users: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def recommend_from_movie(self, movie: str) -> List[str]:
        pass

    @abstractmethod
    def recommend_from_user(self, user: str) -> List[str]:
        pass


class CFBaseline(CFRecommender):
    def __init__(self) -> None:
        super().__init__()

    def train(self, ratings_train: pd.DataFrame) -> None:
        assert {"username", "movie_id", "stars"}.issubset(ratings_train.columns)
        self.ratings_train = ratings_train
        self.users_avg = ratings_train.groupby("username").stars.mean()
        self.movies_avg = ratings_train.groupby("movie_id").stars.mean()

    def predict_ratings(self, movies: List[str], users: List[str]) -> np.ndarray:
        return (self.movies_avg[movies].values + self.users_avg[users].values) / 2

    def recommend_from_movie(self, movie: str) -> List[str]:
        raise NotImplementedError

    def recommend_from_user(self, user: str) -> List[str]:
        raise NotImplementedError


class CFGradFactor(CFRecommender):
    def __init__(
        self,
        all_users: List[str],
        all_movies: List[str],
        num_features: int = 100,
        init_shift: float = 0,
        init_scale: float = 0.5,
        seed: int = 42,
        knn_k_movies: int = 15,
        knn_k_users: int = 100,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.idx2movie = pd.Series(all_movies)
        self.movie2idx = pd.Series(self.idx2movie.index.values, index=self.idx2movie.values)
        self.idx2user = pd.Series(all_users)
        self.user2idx = pd.Series(self.idx2user.index.values, index=self.idx2user.values)
        self.device = device
        self.model = Model(
            num_features, len(self.movie2idx), len(self.user2idx), seed, init_shift, init_scale
        ).to(device)
        self.knn_k_movies = knn_k_movies
        self.knn_k_users = knn_k_users
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.trainer: Optional[Trainer] = None
        self.knn_movies: Optional[NearestNeighbors] = None
        self.knn_users: Optional[NearestNeighbors] = None

    def train(
        self,
        ratings_train: pd.DataFrame,
        ratings_valid: pd.DataFrame,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        patience: int = 3,
        max_epochs: int = 25,
        evaluate_every_n_steps: int = 12500,
        early_stopping_metric: str = "mae_stars",
        early_stopping_threshold: float = 0.02,
        path_to_save: Optional[str] = None,
        return_to_best: bool = True,
    ) -> None:

        assert {"username", "movie_id", "stars"}.issubset(ratings_train.columns)
        assert {"username", "movie_id", "stars"}.issubset(ratings_valid.columns)
        self.ratings_train = ratings_train
        self.ratings_valid = ratings_train

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.trainer = Trainer(self.model, self.optimizer)

        ds_train = RatingDataset(ratings_train, self.movie2idx, self.user2idx)
        ds_valid = RatingDataset(ratings_valid, self.movie2idx, self.user2idx)

        ds_train_mini = Subset(
            ds_train,
            np.random.default_rng(0)
            .choice(len(ds_train), len(ds_valid), replace=False)
            .astype(int)
            .tolist(),
        )

        loader_train = DataLoader(ds_train, batch_size, num_workers=8, prefetch_factor=500)
        loader_valid = DataLoader(ds_valid, batch_size, num_workers=8, prefetch_factor=500)
        loader_train_mini = DataLoader(
            ds_train_mini, batch_size, num_workers=8, prefetch_factor=500
        )

        already_trained = path_to_save is not None and os.path.exists(path_to_save)

        if already_trained:
            self.model.load_state_dict(torch.load(path_to_save, map_location=self.device))  # type: ignore
        else:
            self.trainer.train(
                loader_train,
                loader_train_mini,
                loader_valid,
                patience,
                max_epochs,
                evaluate_every_n_steps,
                early_stopping_metric,
                early_stopping_threshold,
                return_to_best,
            )
            if path_to_save is not None:
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                torch.save(self.model.state_dict(), path_to_save)

        movies_matrix = self.model.movies_weights.weight.clone().detach().cpu().numpy()
        movies_matrix /= np.linalg.norm(movies_matrix, axis=-1, keepdims=True)
        self.knn_movies = NearestNeighbors(n_neighbors=self.knn_k_movies + 1).fit(movies_matrix)

        users_matrix = self.model.users_weights.weight.clone().detach().cpu().numpy()
        users_matrix /= np.linalg.norm(users_matrix, axis=-1, keepdims=True)
        self.knn_users = NearestNeighbors(n_neighbors=self.knn_k_users + 1).fit(users_matrix)

    def predict_ratings(self, movies: List[str], users: List[str]) -> np.ndarray:
        with torch.no_grad():  # type: ignore
            movies_idx = torch.tensor(self.movie2idx[movies].values).to(self.device)
            users_idx = torch.tensor(self.user2idx[users].values).to(self.device)
            return 5 * self.model(movies_idx, users_idx).cpu().numpy()

    def recommend_from_movie(self, movie: str) -> List[str]:
        idx = torch.tensor(self.movie2idx[movie])
        vector = self.model.movies_weights(idx).cpu().numpy().reshape(1, -1)
        assert self.knn_movies is not None
        distances, indices = self.knn_movies.kneighbors(vector)
        return self.idx2movie[indices[0]][1:].tolist()

    def recommend_from_user(self, user: str) -> List[str]:
        idx = torch.tensor(self.user2idx[user])
        vector = self.model.users_weights(idx).cpu().numpy().reshape(1, -1)

        assert self.knn_users is not None
        distances, indices = self.knn_users.kneighbors(vector)
        similar_users = self.idx2user[indices[0]][1:]
        seen_by_user = self.ratings_train.loc[self.ratings_train.username == user].movie_id

        candidates = (
            self.ratings_train.loc[self.ratings_train.username.isin(similar_users)]
            .groupby("movie_id")
            .agg({"stars": ["count", "mean"]})
        )
        candidates.columns = ["_".join(a) for a in candidates.columns.to_flat_index()]

        assert candidates.columns == ["stars_count", "stars_mean"]

        candidates = candidates.loc[~candidates.index.isin(seen_by_user)]
        candidates = candidates.sort_values("stars_count", ascending=False)
        recommended = (
            candidates.loc[candidates.stars_mean > 3.5]
            .head(10)
            .sort_values("stars_mean", ascending=False)
        )
        return recommended.index.tolist()
