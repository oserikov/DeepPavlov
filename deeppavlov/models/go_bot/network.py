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

from logging import getLogger
from typing import Dict, Any, List, Optional, Union, Sequence

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.go_bot.data_handler import DataHandler
from deeppavlov.models.go_bot.features_handling_objects import UtteranceFeatures, DialogueFeatures, BatchDialoguesFeatures
from deeppavlov.models.go_bot.nlu_handler import NLUHandler
from deeppavlov.models.go_bot.nn_stuff_handler import NNStuffHandler
from deeppavlov.models.go_bot.tracker import FeaturizedTracker, DialogueStateTracker, MultipleUserStateTracker
from pathlib import Path

log = getLogger(__name__)

@register("go_bot")
class GoalOrientedBot(NNModel):
    """
    The dialogue bot is based on  https://arxiv.org/abs/1702.03274, which
    introduces Hybrid Code Networks that combine an RNN with domain-specific
    knowledge and system action templates.

    The network handles dialogue policy management.
    Inputs features of an utterance and predicts label of a bot action
    (classification task).

    An LSTM with a dense layer for input features and a dense layer for it's output.
    Softmax is used as an output activation function.

    Todo:
        add docstring for trackers.

    Parameters:
        tokenizer: one of tokenizers from
            :doc:`deeppavlov.models.tokenizers </apiref/models/tokenizers>` module.
        tracker: dialogue state tracker from
            :doc:`deeppavlov.models.go_bot.tracker </apiref/models/go_bot>`.
        hidden_size: size of rnn hidden layer.
        action_size: size of rnn output (equals to number of bot actions).
        dropout_rate: probability of weights dropping out.
        l2_reg_coef: l2 regularization weight (applied to input and output layer).
        dense_size: rnn input size.
        attention_mechanism: describes attention applied to embeddings of input tokens.

            * **type** – type of attention mechanism, possible values are ``'general'``, ``'bahdanau'``,
              ``'light_general'``, ``'light_bahdanau'``, ``'cs_general'`` and ``'cs_bahdanau'``.
            * **hidden_size** – attention hidden state size.
            * **max_num_tokens** – maximum number of input tokens.
            * **depth** – number of averages used in constrained attentions
              (``'cs_bahdanau'`` or ``'cs_general'``).
            * **action_as_key** – whether to use action from previous timestep as key
              to attention.
            * **intent_as_key** – use utterance intents as attention key or not.
            * **projected_align** – whether to use output projection.
        network_parameters: dictionary with network parameters (for compatibility with release 0.1.1,
            deprecated in the future)

        template_path: file with mapping between actions and text templates
            for response generation.
        template_type: type of used response templates in string format.
        word_vocab: vocabulary of input word tokens
            (:class:`~deeppavlov.core.data.simple_vocab.SimpleVocabulary` recommended).
        bow_embedder: instance of one-hot word encoder
            :class:`~deeppavlov.models.embedders.bow_embedder.BoWEmbedder`.
        embedder: one of embedders from
            :doc:`deeppavlov.models.embedders </apiref/models/embedders>` module.
        slot_filler: component that outputs slot values for a given utterance
            (:class:`~deeppavlov.models.slotfill.slotfill.DstcSlotFillingNetwork`
            recommended).
        intent_classifier: component that outputs intents probability
            distribution for a given utterance (
            :class:`~deeppavlov.models.classifiers.keras_classification_model.KerasClassificationModel`
            recommended).
        database: database that will be used during inference to perform
            ``api_call_action`` actions and get ``'db_result'`` result (
            :class:`~deeppavlov.core.data.sqlite_database.Sqlite3Database`
            recommended).
        api_call_action: label of the action that corresponds to database api call
            (it must be present in your ``template_path`` file), during interaction
            it will be used to get ``'db_result'`` from ``database``.
        use_action_mask: if ``True``, network output will be applied with a mask
            over allowed actions.
        debug: whether to display debug output.
    """

    def __init__(self,
                 tokenizer: Component,
                 tracker: FeaturizedTracker,
                 template_path: str,
                 save_path: str,
                 hidden_size: int = 128,
                 action_size: int = None,
                 dropout_rate: float = 0.,
                 l2_reg_coef: float = 0.,
                 dense_size: int = None,
                 attention_mechanism: dict = None,
                 network_parameters: Optional[Dict[str, Any]] = None,
                 load_path: str = None,
                 template_type: str = "DefaultTemplate",
                 word_vocab: Component = None,
                 bow_embedder: Component = None,
                 embedder: Component = None,
                 slot_filler: Component = None,
                 intent_classifier: Component = None,
                 database: Component = None,
                 api_call_action: str = None,
                 use_action_mask: bool = False,  # todo not supported actually
                 debug: bool = False,
                 **kwargs) -> None:

        super().__init__(save_path=save_path, load_path=load_path, **kwargs)

        # todo tracker params dto; data_config_dto; method_name

        self.debug = debug

        self.nlu_handler = NLUHandler(tokenizer, slot_filler, intent_classifier)
        self.data_handler = DataHandler(debug, template_path, template_type, word_vocab, bow_embedder, api_call_action, embedder)

        n_actions = len(self.data_handler.templates)

        self.dialogue_state_tracker = DialogueStateTracker(tracker.slot_names, n_actions, hidden_size, database)
        self.multiple_user_state_tracker = MultipleUserStateTracker()

        embedder_dim = self.data_handler.embedder.dim if self.data_handler.embedder else None
        use_bow_embedder = self.data_handler.use_bow_encoder()
        word_vocab_size = self.data_handler.word_vocab_size()
        nn_stuff_save_path = Path(save_path, NNStuffHandler.SAVE_LOAD_SUBDIR_NAME)
        nn_stuff_load_path = Path(load_path, NNStuffHandler.SAVE_LOAD_SUBDIR_NAME)

        self.policy = NNStuffHandler(
            hidden_size, action_size, dropout_rate, l2_reg_coef, dense_size, attention_mechanism,
            network_parameters, embedder_dim, n_actions,
            self.intent_classifier, self.intents, tracker.num_features,
            use_bow_embedder, word_vocab_size,
            load_path=nn_stuff_load_path, save_path=nn_stuff_save_path,
            **kwargs)

        self.reset()  # tracker

    def calc_dialogues_batches_training_data(self, x: List[dict], y: List[dict]) -> BatchDialoguesFeatures:
        max_num_utter = max(len(d_contexts) for d_contexts in x)  # for padding
        batch_features = BatchDialoguesFeatures(max_num_utter)

        for d_contexts, d_responses in zip(x, y):
            dialogue_features = self.calc_dialogue_training_data(d_contexts, d_responses)
            batch_features.append(dialogue_features)

        return batch_features

    def calc_dialogue_training_data(self, d_contexts, d_responses) -> DialogueFeatures:
        dialogue_features = DialogueFeatures()
        self.dialogue_state_tracker.reset_state()
        for context, response in zip(d_contexts, d_responses):
            utterance_features = self.process_utterance(context, response)

            dialogue_features.append(utterance_features)

            self.dialogue_state_tracker.update_previous_action(utterance_features.action_id)

            if self.debug:
                log.debug(f"True response = '{response['text']}'.")
                if utterance_features.action_mask[utterance_features.action_id] != 1.:
                    log.warning("True action forbidden by action mask.")
        return dialogue_features

    def process_utterance(self, context, response) -> UtteranceFeatures:
        text = context['text']

        self.dialogue_state_tracker.update_ground_truth_db_result_from_context(context)
        attn_key, features, tokens_embeddings_padded, action_mask = self.method_name(text, self.dialogue_state_tracker)

        action_id = self.data_handler.encode_response(response['act'])

        utterance_features = UtteranceFeatures(action_id=action_id,
                                               action_mask=action_mask,
                                               attn_key=attn_key,
                                               features=features,
                                               tokens_embeddings_padded=tokens_embeddings_padded)
        return utterance_features

    def method_name(self, text, tracker, keep_tracker_state=False):
        context_slots, intent_features, tokens = self.nlu(text)

        # region text2vec
        tokens_bow_encoded = []
        if self.data_handler.use_bow_encoder():
            tokens_bow_encoded = self.data_handler.bow_encode_tokens(tokens)

        tokens_embeddings_padded = np.array([], dtype=np.float32)
        tokens_aggregated_embedding = []
        if self.policy.get_attn_hyperparams():
            attn_window_size = self.policy.get_attn_hyperparams().window_size
            tokens_embeddings_padded = self.data_handler.calc_tokens_embeddings(attn_window_size, tokens)
        else:
            tokens_aggregated_embedding = self.data_handler.calc_tokens_embedding(tokens)
        # endregion text2vec

        if context_slots and not keep_tracker_state:
            tracker.update_state(context_slots)

        # todo tracker features DTO
        tracker_prev_action = tracker.prev_action
        state_features = tracker.get_features()

        # region features engineering
        context_features = self.calc_context_features(tracker)

        attn_key = self.calc_attn_key(self.policy.get_attn_hyperparams(), intent_features, tracker_prev_action)

        concat_feats = np.hstack((tokens_bow_encoded, tokens_aggregated_embedding, intent_features, state_features, context_features, tracker_prev_action))
        # endregion features engineering

        action_mask = tracker.calc_action_mask(self.data_handler.api_call_id)

        return attn_key, concat_feats, tokens_embeddings_padded, action_mask

    # todo как инфер понимает из конфига что ему нужно. лёша что-то говорил про дерево
    def _infer(self, text: str, tracker: DialogueStateTracker, keep_tracker_state=False) -> Sequence:
        attn_key, concat_feats, tokens_embeddings_padded, action_mask = self.method_name(text, tracker, keep_tracker_state)

        utterance_features = UtteranceFeatures(action_id=None,
                                               action_mask=action_mask,
                                               attn_key=attn_key,
                                               features=concat_feats,
                                               tokens_embeddings_padded=tokens_embeddings_padded)

        tracker_net_state_c, tracker_net_state_h = tracker.network_state[0], tracker.network_state[1]
        probs, state_c, state_h = self.policy._network_call(utterance_features, tracker_net_state_c, tracker_net_state_h, prob=True)  # todo чо за warning кидает ide, почему

        return probs, np.argmax(probs), (state_c, state_h)

    def __call__(self, batch: Union[List[dict], List[str]], user_ids: Optional[List] = None) -> Union[List[str],
                                                                                                      List[List[str]]]:
        # batch is a list of utterances
        if isinstance(batch[0], str):
            res = []
            if not user_ids:
                user_ids = ['finn'] * len(batch)
            for user_id, x in zip(user_ids, batch):
                if not self.multiple_user_state_tracker.check_new_user(user_id):
                    self.multiple_user_state_tracker.init_new_tracker(user_id, self.dialogue_state_tracker)

                tracker = self.multiple_user_state_tracker.get_user_tracker(user_id)

                _, predicted_act_id, tracker.network_state = self._infer(x, tracker=tracker)
                tracker.update_previous_action(predicted_act_id)

                if predicted_act_id == self.data_handler.api_call_id:
                    tracker.make_api_call()
                    _, predicted_act_id, tracker.network_state = self._infer(x, tracker=tracker, keep_tracker_state=True)
                    tracker.update_previous_action(predicted_act_id)

                # region nlg
                resp = self.data_handler.decode_response(predicted_act_id, tracker)
                # endregion nlg
                res.append(resp)
            return res
        # batch is a list of dialogs, user_ids ignored
        # todo: что значит коммент выше, почему узер идс игноред
        return [self._infer_dialog(x) for x in batch]

    def _infer_dialog(self, contexts: List[dict]) -> List[str]:
        res = []
        self.dialogue_state_tracker.reset_state()
        for context in contexts:
            if context.get('prev_resp_act') is not None:
                previous_act_id = self.data_handler.encode_response(context['prev_resp_act'])
                self.dialogue_state_tracker.update_previous_action(previous_act_id)  # teacher-forcing

            self.dialogue_state_tracker.update_ground_truth_db_result_from_context(context)

            text = context['text']
            _, predicted_act_id, self.dialogue_state_tracker.network_state = self._infer(text, tracker=self.dialogue_state_tracker)
            self.dialogue_state_tracker.update_previous_action(predicted_act_id)

            resp = self.data_handler.decode_response(predicted_act_id, self.dialogue_state_tracker)

            res.append(resp)
        return res

    def train_on_batch(self, x: List[dict], y: List[dict]) -> dict:
        batch_features = self.calc_dialogues_batches_training_data(x, y)
        return self.policy._network_train_on_batch(batch_features)

    def reset(self, user_id: Union[None, str, int] = None) -> None:
        # todo а чо, у нас всё что можно закешить лежит в мультиюхертрекере?
        self.multiple_user_state_tracker.reset(user_id)
        if self.debug:
            log.debug("Bot reset.")

    def load(self, *args, **kwargs) -> None:
        self.policy.load()
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        super().save(*args, **kwargs)
        self.policy.save()