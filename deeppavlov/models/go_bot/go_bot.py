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
from deeppavlov.models.go_bot.dto.dataset_features import UtteranceDataEntry, DialogueDataEntry, \
    BatchDialoguesDataset, UtteranceFeatures
from deeppavlov.models.go_bot.features_engineerer import FeaturesParams
from deeppavlov.models.go_bot.nlg_mechanism import NLGHandler
from deeppavlov.models.go_bot.nlu_mechanism import NLUHandler
from deeppavlov.models.go_bot.policy import PolicyNetwork, PolicyNetworkParams
from deeppavlov.models.go_bot.tracker import FeaturizedTracker, DialogueStateTracker, MultipleUserStateTrackersPool
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

    DEFAULT_USER_ID = 1
    POLICY_DIR_NAME = "policy"

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
                 use_action_mask: bool = False,
                 debug: bool = False,
                 **kwargs) -> None:
        self.use_action_mask = use_action_mask  # todo not supported actually
        super().__init__(save_path=save_path, load_path=load_path, **kwargs)

        self.debug = debug

        policy_network_params = PolicyNetworkParams(hidden_size, action_size, dropout_rate, l2_reg_coef,
                                                    dense_size, attention_mechanism, network_parameters)

        self.nlu_handler = NLUHandler(tokenizer, slot_filler, intent_classifier)
        self.nlg_handler = NLGHandler(template_path, template_type, api_call_action)
        self.data_handler = DataHandler(debug, word_vocab, bow_embedder, embedder)

        self.dialogue_state_tracker = DialogueStateTracker.from_gobot_params(tracker, self.nlg_handler, policy_network_params, database)
        self.multiple_user_state_tracker = MultipleUserStateTrackersPool(base_tracker=self.dialogue_state_tracker)

        tokens_dims = self.data_handler.get_dims()
        features_params = FeaturesParams.from_configured(self.nlg_handler, self.nlu_handler, self.dialogue_state_tracker)
        policy_save_path = Path(save_path, self.POLICY_DIR_NAME)
        policy_load_path = Path(load_path, self.POLICY_DIR_NAME)

        self.policy = PolicyNetwork(policy_network_params, tokens_dims, features_params,
                                    policy_load_path, policy_save_path, **kwargs)

        self.reset()

    def calc_dialogues_batches_training_data(self, x: List[dict], y: List[dict]) -> BatchDialoguesDataset:
        max_num_utter = max(len(d_contexts) for d_contexts in x)  # for padding
        batch_features = BatchDialoguesDataset(max_num_utter)

        for d_contexts, d_responses in zip(x, y):
            dialogue_features = self.calc_dialogue_training_data(d_contexts, d_responses)
            batch_features.append(dialogue_features)

        return batch_features

    def calc_dialogue_training_data(self, d_contexts, d_responses) -> DialogueDataEntry:
        dialogue_features = DialogueDataEntry()
        self.dialogue_state_tracker.reset_state()
        for context, response in zip(d_contexts, d_responses):
            utterance_features = self.process_utterance(context, response)

            dialogue_features.append(utterance_features)

            self.dialogue_state_tracker.update_previous_action(utterance_features.target.action_id)

            if self.debug:
                log.debug(f"True response = '{response['text']}'.")
                if utterance_features.features.action_mask[utterance_features.target.action_id] != 1.:
                    log.warning("True action forbidden by action mask.")
        return dialogue_features

    def process_utterance(self, context, response) -> UtteranceDataEntry:
        text = context['text']

        self.dialogue_state_tracker.update_ground_truth_db_result_from_context(context)
        attn_key, features, tokens_embeddings_padded, action_mask = self.method_name(text, self.dialogue_state_tracker)

        action_id = self.nlg_handler.encode_response(response['act'])

        utterance_features = UtteranceDataEntry(action_id=action_id,
                                                action_mask=action_mask,
                                                attn_key=attn_key,
                                                features=features,
                                                tokens_embeddings_padded=tokens_embeddings_padded)
        return utterance_features

    def method_name(self, text, tracker, keep_tracker_state=False) -> UtteranceFeatures:
        context_slots, intent_features, tokens = self.nlu(text)

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

        if context_slots and not keep_tracker_state:
            tracker.update_state(context_slots)

        # todo tracker features DTO
        tracker_prev_action = tracker.prev_action
        state_features = tracker.get_features()

        # region features engineering
        context_features = self.calc_context_features(tracker)

        attn_key = self.calc_attn_key(self.policy.get_attn_hyperparams(), intent_features, tracker_prev_action)

        concat_feats = np.hstack((tokens_bow_encoded, tokens_aggregated_embedding, intent_features, state_features,
                                  context_features, tracker_prev_action))
        # endregion features engineering

        action_mask = tracker.calc_action_mask(self.nlg_handler.api_call_id)

        return UtteranceFeatures(action_mask, attn_key, tokens_embeddings_padded, concat_feats)

    # todo как инфер понимает из конфига что ему нужно. лёша что-то говорил про дерево
    def _infer(self, text: str, tracker: DialogueStateTracker, keep_tracker_state=False) -> Sequence:
        attn_key, concat_feats, tokens_embeddings_padded, action_mask = self.method_name(text, tracker,
                                                                                         keep_tracker_state)

        utterance_data_entry = UtteranceDataEntry(action_id=None,
                                                  action_mask=action_mask,
                                                  attn_key=attn_key,
                                                  features=concat_feats,
                                                  tokens_embeddings_padded=tokens_embeddings_padded)

        dialogue_data_entry = DialogueDataEntry()
        dialogue_data_entry.append(utterance_data_entry)
        batch_dialogues_dataset = BatchDialoguesDataset(1)
        batch_dialogues_dataset.append(dialogue_data_entry)

        tracker_net_state_c, tracker_net_state_h = tracker.network_state[0], tracker.network_state[1]
        probs, state_c, state_h = self.policy.__call__(batch_dialogues_dataset.features,
                                                       tracker_net_state_c,
                                                       tracker_net_state_h,
                                                       prob=True)  # todo чо за warning кидает ide, почему

        return probs, np.argmax(probs), (state_c, state_h)

    def __call__(self, batch: Union[List[List[dict]], List[str]], user_ids: Optional[List] = None) -> Union[List[str],
                                                                                                      List[List[str]]]:
        # todo refactor
        # batch is a list of utterances
        if isinstance(batch[0], str):
            res = []
            if not user_ids:
                user_ids = [self.DEFAULT_USER_ID] * len(batch)
            for user_id, x in zip(user_ids, batch):
                x: str

                tracker = self.multiple_user_state_tracker.get_or_init_tracker(user_id)

                _, predicted_act_id, network_state = self._infer(x, tracker)
                tracker.update_previous_action(predicted_act_id)
                tracker.network_state = network_state

                if predicted_act_id == self.nlg_handler.api_call_id:
                    tracker.make_api_call()
                    _, predicted_act_id, network_state = self._infer(x, tracker, keep_tracker_state=True)
                    tracker.update_previous_action(predicted_act_id)
                    tracker.network_state = network_state

                tracker_slotfilled_state = tracker.fill_current_state_with_db_results()
                resp = self.nlg_handler.generate_slotfilled_text_for_action(predicted_act_id, tracker_slotfilled_state)

                res.append(resp)
        else:
            # batch is a list of dialogs, user_ids ignored
            res = [self._infer_dialog(x) for x in batch]
        return res

    def _infer_dialog(self, contexts: List[dict]) -> List[str]:
        res = []
        self.dialogue_state_tracker.reset_state()
        for context in contexts:
            if context.get('prev_resp_act') is not None:
                previous_act_id = self.nlg_handler.encode_response(context['prev_resp_act'])
                self.dialogue_state_tracker.update_previous_action(previous_act_id)

            self.dialogue_state_tracker.update_ground_truth_db_result_from_context(context)

            _, predicted_act_id, network_state = self._infer(context['text'], self.dialogue_state_tracker)
            self.dialogue_state_tracker.update_previous_action(predicted_act_id)
            self.dialogue_state_tracker.network_state = network_state

            tracker_slotfilled_state = self.dialogue_state_tracker.fill_current_state_with_db_results()
            resp = self.nlg_handler.generate_slotfilled_text_for_action(predicted_act_id, tracker_slotfilled_state)

            res.append(resp)
        return res

    def train_on_batch(self, x: List[dict], y: List[dict]) -> dict:
        batch_features_dataset = self.calc_dialogues_batches_training_data(x, y)
        batch_features, batch_targets = batch_features_dataset.features, batch_features_dataset.targets
        return self.policy.train_on_batch(batch_features, batch_targets)

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
