# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle

from logging import getLogger
from typing import Iterator

import numpy as np
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.models.embedders.abstract_embedder import Embedder

log = getLogger(__name__)


@register('word2vec')
class Word2VecEmbedder(Embedder):
    def _get_word_vector(self, w: str) -> np.ndarray:
        return self.model.get(w, self.zero_vec)

    def load(self) -> None:
        log.info(f"[loading Word2Vec embeddings from `{self.load_path}`]")
        self.model = pickle.load(str(self.load_path))
        self.zero_vec = np.zeros(300, dtype=float)
        self.dim = 300

    @overrides
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all words from fastText model vocabulary

        Returns:
            iterator
        """
        yield from self.model.keys()
