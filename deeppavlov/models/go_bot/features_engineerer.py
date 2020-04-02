import numpy as np

from deeppavlov.models.go_bot.nlg_mechanism import NLGHandler
from deeppavlov.models.go_bot.nlu_mechanism import NLUHandler
from deeppavlov.models.go_bot.tracker import DialogueStateTracker, FeaturizedTracker


class FeaturesParams:

    num_actions: int
    num_intents: int
    num_tracker_features: int

    def __init__(self, num_actions, num_intents, num_tracker_features):
        self.num_actions = num_actions
        self.num_intents = num_intents
        self.num_tracker_features = num_tracker_features

    @staticmethod
    def from_configured(nlg_handler: NLGHandler, nlu_handler: NLUHandler, tracker: FeaturizedTracker):
        return FeaturesParams(nlg_handler.num_of_known_actions(),
                              nlu_handler.num_of_known_intents(),
                              tracker.num_features)

class FeaturesEngineer:
    @staticmethod
    def calc_context_features(tracker: DialogueStateTracker):
        # todo seems like itneeds to be moved to tracker
        current_db_result = tracker.current_db_result
        db_result = tracker.db_result
        dst_state = tracker.get_state()

        result_matches_state = 0.
        if current_db_result is not None:
            matching_items = dst_state.items()
            result_matches_state = all(v == db_result.get(s)
                                       for s, v in matching_items
                                       if v != 'dontcare') * 1.
        context_features = np.array([
            bool(current_db_result) * 1.,
            (current_db_result == {}) * 1.,
            (db_result is None) * 1.,
            bool(db_result) * 1.,
            (db_result == {}) * 1.,
            result_matches_state
        ], dtype=np.float32)
        return context_features

    @staticmethod
    def calc_attn_key(attn_hyperparams, intent_features, tracker_prev_action):
        # todo to attn mechanism
        attn_key = np.array([], dtype=np.float32)

        if attn_hyperparams:
            if attn_hyperparams.use_action_as_key:
                attn_key = np.hstack((attn_key, tracker_prev_action))
            if attn_hyperparams.use_intent_as_key:
                attn_key = np.hstack((attn_key, intent_features))
            if len(attn_key) == 0:
                attn_key = np.array([1], dtype=np.float32)
        return attn_key