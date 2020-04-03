from deeppavlov.models.go_bot.data_handler import TokensVectorRepresentationParams
from deeppavlov.models.go_bot.features_engineerer import FeaturesParams


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

    @staticmethod
    def configure_attn(attn,
                       tokens_dims: TokensVectorRepresentationParams,
                       features_params: FeaturesParams):
        curr_attn_token_size = attn.get('token_size')
        curr_attn_action_as_key = attn.get('action_as_key')
        curr_attn_intent_as_key = attn.get('intent_as_key')
        curr_attn_key_size = attn.get('key_size')

        token_size = curr_attn_token_size or tokens_dims.embedding_dim
        action_as_key = curr_attn_action_as_key or False
        intent_as_key = curr_attn_intent_as_key or False

        possible_key_size = 0
        if action_as_key:
            possible_key_size += features_params.num_actions
        if intent_as_key and features_params.num_intents:
            possible_key_size += features_params.num_intents
        possible_key_size = possible_key_size or 1
        key_size = curr_attn_key_size or possible_key_size

        gobot_attn_params = GobotAttnParams(max_num_tokens=attn.get("max_num_tokens"),
                                            hidden_size=attn.get("hidden_size"),
                                            token_size=token_size,
                                            key_size=key_size,
                                            type_=attn.get("type"),
                                            projected_align=attn.get("projected_align"),
                                            depth=attn.get("depth"),
                                            action_as_key=action_as_key,
                                            intent_as_key=intent_as_key)

        return gobot_attn_params


class GobotAttnMechanism:
    # todo migrate to dataclasses?
    key_size: int
    token_size: int
    window_size: int
    use_action_as_key: bool
    use_intent_as_key: bool

    def __init__(self, attn_params: GobotAttnParams):
        self.key_size = attn_params.key_size
        self.token_size = attn_params.token_size
        self.window_size = attn_params.max_num_tokens
        self.use_action_as_key = attn_params.action_as_key
        self.use_intent_as_key = attn_params.intent_as_key
