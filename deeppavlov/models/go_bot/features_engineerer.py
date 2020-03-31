import numpy as np


class FeaturesEngineer:
    @staticmethod
    def calc_context_features(current_db_result, db_result, dst_state):
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
        attn_key = np.array([], dtype=np.float32)

        if attn_hyperparams:
            if attn_hyperparams.use_action_as_key:
                attn_key = np.hstack((attn_key, tracker_prev_action))
            if attn_hyperparams.use_intent_as_key:
                attn_key = np.hstack((attn_key, intent_features))
            if len(attn_key) == 0:
                attn_key = np.array([1], dtype=np.float32)
        return attn_key