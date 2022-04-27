import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.nn.parameter import Parameter


class Model(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_movies: int,
        num_users: int,
        seed: int,
        shift: float,
        scale: float,
    ):
        super().__init__()
        random_gen = np.random.default_rng(seed)

        c = 0.6
        self.movies_weights = Embedding.from_pretrained(
            torch.from_numpy(
                random_gen.normal(shift, scale, (num_movies, num_features)).clip(-c, c)
            ).float()
        )  # type: ignore

        self.users_weights = Embedding.from_pretrained(
            torch.from_numpy(
                random_gen.normal(shift, scale, (num_users, num_features)).clip(-c, c)
            ).float()
        )  # type: ignore

        self.movies_biases = Parameter(
            torch.from_numpy(random_gen.normal(0, scale, (num_movies,)).clip(-c, c)).float()
        )

        self.users_biases = Parameter(
            torch.from_numpy(random_gen.normal(0, scale, (num_users,)).clip(-c, c)).float()
        )

    def forward(self, movie_idx: Tensor, user_idx: Tensor) -> Tensor:
        m_w = self.movies_weights(movie_idx)
        m_b = self.movies_biases[movie_idx]
        u_w = self.users_weights(user_idx)
        u_b = self.users_biases[user_idx]
        return torch.sigmoid((m_w * u_w).sum(-1) + m_b + u_b)
