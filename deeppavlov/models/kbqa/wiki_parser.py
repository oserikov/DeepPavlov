from pathlib import Path
from hdt import HDTDocument
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

@register('wiki_parser')
class WikiParser(Component):
    def __init__(self, wiki_filename, **kwargs):
        wiki_path = expand_path(wiki_filename)
        self.document = HDTDocument(str(wiki_path))

    def __call__(self, what_return, direction, entity, rel=None, obj=None, type_of_rel=None, filter_obj=None):
        entity = "http://www.wikidata.org/entity/"+entity
        if rel is not None:
            rel = "http://www.wikidata.org/prop/{}/{}".format(type_of_rel, rel)
        else:
            rel = ""
        
        if obj is None:
            obj = ""

        if direction == "forw":
            triplets, cardinality = self.document.search_triples(entity, rel, obj)
        if direction == "backw":
            triplets, cardinality = self.document.search_triples(obj, rel, entity)

        found_triplets = []
        for triplet in triplets:
            if type_of_rel is None or (type_of_rel is not None and type_of_rel in triplet[1]):
                if filter_obj is None or (filter_obj is not None and filter_obj in triplet[2]):
                    found_triplets.append(triplet)

        if what_return == "rels":
            rels = [triplet[1].split('/')[-1] for triplet in found_triplets]
            rels = list(set(rels))
            return rels
        
        if what_return == "triplets":
            return found_triplets

        if what_return == "objects":
            if direction == "forw":
                objects = [triplet[2] for triplet in found_triplets]
            if direction == "backw":
                objects = [triplet[0] for triplet in found_triplets]
            return objects

