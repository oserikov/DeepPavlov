from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    def __init__(self, thres_proba=0.86, **kwargs):
        self.thres_proba = thres_proba
        pass

    def __call__(self, question_tokens, token_probas, **kwargs):
        tokens, probas = question_tokens[0], token_probas[0]

        tags = []
        for proba in probas:
            if proba[0] <= self.thres_proba:
                tags.append(1)
            if proba[0] > self.thres_proba:
                tags.append(0)

        entities = self.entities_from_tags(tokens, tags, probas)
        return entities

    def entities_from_tags(self, tokens, tags, probas):
        entities = []
        start = 0
        entity = ''

        for tok, tag, proba in zip(tokens, tags, probas):
            if tag != 0 and start == 0:
                start = 1
                entity = tok
            elif tag != 0 and start == 1:
                entity += ' '
                entity += tok
            elif tag == 0 and len(entity) > 0 and start == 1:
                start = 0
                entities.append(entity.replace(' - ', '-').replace("'s", '').replace(' .','').replace('{', '').replace('}', ''))
            else:
                pass

        return entities

