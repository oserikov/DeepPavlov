import itertools
import pickle
from logging import getLogger
from typing import List, Dict, Tuple, Optional

import nltk
import pymorphy2
from fuzzywuzzy import fuzz

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher

log = getLogger(__name__)


@register('entity_linker_cq')
class EntityLinkerCQ(Component, Serializable):

    def __init__(self, load_path: str, inverted_index_filename: str, id_to_name_file: str,
                       use_prefix_tree: bool = False, debug: bool = False, *args, **kwargs) -> None:
        
        super().__init__(save_path=None, load_path=load_path)
        self.use_prefix_tree = use_prefix_tree
        self.debug = debug

        self.inverted_index_filename = inverted_index_filename
        self.id_to_name_file = id_to_name_file
        self.inverted_index: Optional[Dict[str, List[Tuple[str]]]] = None
        self.id_to_name: Optional[Dict[str, Dict[List[str]]]] = None
        self.load()

        if self.use_prefix_tree:
            alphabet = "!#%\&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz½¿ÁÄÅÆÇÉÎÓÖ×ÚßàáâãäåæçèéêëíîïðñòóôöøùúûüýāăąćČčĐėęěĞğĩīİıŁłńňŌōőřŚśşŠšťũūůŵźŻżŽžơưșȚțəʻʿΠΡβγБМавдежикмностъяḤḥṇṬṭầếờợ–‘’Ⅲ−∗"
            dictionary_words = list(self.inverted_index.keys())
            self.searcher = LevenshteinSearcher(alphabet, dictionary_words)

    def load(self) -> None:
        with open(self.load_path / self.inverted_index_filename, 'rb') as inv:
            self.inverted_index = pickle.load(inv)
            self.inverted_index: Dict[str, List[Tuple[str]]]
        with open(self.load_path / self.id_to_name_file, 'rb') as i2n:
            self.id_to_name = pickle.load(i2n)
            self.id_to_name: Dict[str, Dict[List[str]]]

    def save(self) -> None:
        pass

    def __call__(self, entity):
        confidences = []
        srtd_cand_ent = []
        if not entity:
            wiki_entities = ['None']
        else:
            candidate_entities = self.candidate_entities_inverted_index(entity)
            candidate_names = self.candidate_entities_names(candidate_entities)
            wiki_entities, confidences, srtd_cand_ent = self.sort_found_entities(candidate_entities,
                                                                                 candidate_names, entity)

        return wiki_entities

    def candidate_entities_inverted_index(self, entity: str) -> List[Tuple[str]]:
        word_tokens = nltk.word_tokenize(entity)
        candidate_entities = []

        for tok in word_tokens:
            if len(tok) > 1:
                found = False
                if tok in self.inverted_index:
                    candidate_entities += self.inverted_index[tok]
                    found = True
                
                if not found and self.use_prefix_tree:
                    words_with_levens_1 = self.searcher.search(tok, d=1)
                    for word in words_with_levens_1:
                        candidate_entities += self.inverted_index[word[0]]
        candidate_entities = list(set(candidate_entities))

        return candidate_entities

    def sort_found_entities(self, candidate_entities: List[Tuple[str]],
                            candidate_names: List[List[str]],
                            entity: str) -> Tuple[List[str], List[str], List[Tuple[str]]]:
        entities_ratios = []
        for candidate, entity_names in zip(candidate_entities, candidate_names):
            entity_id = candidate[0]
            num_rels = candidate[1]
            entity_name = entity_names[0]
            fuzz_ratio = max([fuzz.ratio(name.lower(), entity.lower()) for name in entity_names]) 
            entities_ratios.append((entity_name, entity_id, fuzz_ratio, num_rels))

        srtd_with_ratios = sorted(entities_ratios, key=lambda x: (x[2], x[3]), reverse=True)
        wiki_entities = [ent[1] for ent in srtd_with_ratios]
        confidences = [float(ent[2]) * 0.01 for ent in srtd_with_ratios]

        return wiki_entities, confidences, srtd_with_ratios

    def candidate_entities_names(self, candidate_entities: List[Tuple[str]]) -> List[List[str]]:
        candidate_names = []
        for candidate in candidate_entities:
            entity_id = candidate[0]
            entity_names = [self.id_to_name[entity_id]["name"]]
            if "aliases" in self.id_to_name[entity_id].keys():
                aliases = self.id_to_name[entity_id]["aliases"]
                for alias in aliases:
                    entity_names.append(alias)
            candidate_names.append(entity_names)

        return candidate_names

