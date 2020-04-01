from typing import Sequence, NamedTuple, List

import numpy as np


class UtteranceFeatures:
    action_id: int
    action_mask: np.ndarray[float]
    attn_key: np.ndarray[float]
    tokens_embeddings_padded: np.ndarray[float]
    features: np.ndarray

    def __init__(self, action_id, action_mask, attn_key, tokens_embeddings_padded, features):
        self.action_id = action_id
        self.action_mask = action_mask
        self.attn_key = attn_key
        self.tokens_embeddings_padded = tokens_embeddings_padded
        self.features = features

class DialogueFeatures:
    action_ids: List[int]
    action_masks: List[np.ndarray[float]]
    attn_keys: List[np.ndarray[float]]
    tokens_embeddings_paddeds: List[np.ndarray[float]]
    featuress: List[np.ndarray]

    def __init__(self):
        self.action_ids = []
        self.action_masks = []
        self.attn_keys = []
        self.tokens_embeddings_paddeds = []
        self.featuress = []

    def append(self, utterance_features: UtteranceFeatures):
        self.action_ids.append(utterance_features.action_id)
        self.action_masks.append(utterance_features.action_mask)
        self.attn_keys.append(utterance_features.attn_key)
        self.tokens_embeddings_paddeds.append(utterance_features.tokens_embeddings_padded)
        self.featuress.append(utterance_features.features)

    def __len__(self):
        return len(self.featuress)

class PaddedDialogueFeatures(DialogueFeatures):
    padded_dialogue_length_mask: List[int]

    def __init__(self, dialogue_features: DialogueFeatures, sequence_length):
        super().__init__()

        padding_length = sequence_length - len(dialogue_features)

        self.action_ids = dialogue_features.action_ids + [0] * padding_length
        self.action_masks = dialogue_features.action_masks + [np.zeros_like(dialogue_features.action_masks[0])] * padding_length
        self.attn_keys = dialogue_features.attn_keys + [np.zeros_like(dialogue_features.attn_keys[0])] * padding_length
        self.tokens_embeddings_paddeds = dialogue_features.tokens_embeddings_paddeds + [np.zeros_like(dialogue_features.tokens_embeddings_paddeds[0])] * padding_length
        self.featuress = dialogue_features.featuress + [np.zeros_like(dialogue_features.featuress[0])] * padding_length

        self.padded_dialogue_length_mask = [1] * len(dialogue_features) + [0] * padding_length


class BatchDialoguesFeatures:
    b_action_ids: List[List[int]]
    b_action_masks: List[List[np.ndarray[float]]]
    b_attn_keys: List[List[np.ndarray[float]]]
    b_tokens_embeddings_paddeds: List[List[np.ndarray[float]]]
    b_featuress: List[List[np.ndarray]]
    b_padded_dialogue_length_mask: List[List[int]]

    def __init__(self, max_dialogue_length):
        self.b_action_ids = []
        self.b_action_masks = []
        self.b_attn_keys = []
        self.b_tokens_embeddings_paddeds = []
        self.b_featuress = []
        self.b_padded_dialogue_length_mask = []
        self.max_dialogue_length = max_dialogue_length

    def append(self, dialogue_features: DialogueFeatures):
        padded_dialogue_features = PaddedDialogueFeatures(dialogue_features, self.max_dialogue_length)
        self.b_action_ids.append(padded_dialogue_features.action_ids)
        self.b_action_masks.append(padded_dialogue_features.action_masks)
        self.b_attn_keys.append(padded_dialogue_features.attn_keys)
        self.b_tokens_embeddings_paddeds.append(padded_dialogue_features.tokens_embeddings_paddeds)
        self.b_featuress.append(padded_dialogue_features.featuress)
        self.b_padded_dialogue_length_mask.append(padded_dialogue_features.padded_dialogue_length_mask)

    def __len__(self):
        return len(self.b_featuress)