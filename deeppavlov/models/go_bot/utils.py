from typing import NamedTuple

class GobotAttnParams:
    max_num_tokens: int
    hidden_size: int
    token_size: int
    key_size: int
    type: str
    projected_align: bool
    depth: int
    action_as_key: bool
    intent_as_key: bool

    def __init__(self,max_num_tokens, hidden_size, token_size, key_size, type_, projected_align, depth, action_as_key, intent_as_key):
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.token_size = token_size
        self.key_size = key_size
        self.type = type_
        self.projected_align = projected_align
        self.depth = depth
        self.action_as_key = action_as_key
        self.intent_as_key = intent_as_key


class GobotAttnHyperParams:
    # todo migrate to dataclasses?
    key_size: int
    token_size: int
    window_size: int
    use_action_as_key: bool
    use_intent_as_key: bool

    def __init__(self, attn_params: GobotAttnParams):
        self.key_size = attn_params.key_size
        self.token_size = attn_params.token_size
        self.window_size = attn_params.window_size
        self.use_action_as_key = attn_params.action_as_key
        self.use_intent_as_key = attn_params.intent_as_key
