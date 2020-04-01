import collections
import json
from typing import Tuple, List, Optional, Sequence
from logging import getLogger

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.layers import tf_attention_mechanisms as am, tf_layers
from tensorflow.contrib.layers import xavier_initializer as xav

from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.go_bot.features_handling_objects import BatchDialoguesFeatures, UtteranceFeatures, \
    DialogueFeatures
from deeppavlov.models.go_bot.utils import GobotAttnHyperParams, GobotAttnParams


def calc_obs_size(default_tracker_num_features,
                  n_actions,
                  use_bow_embedder, word_vocab_size, embedder_dim,
                  intent_classifier, intents):
    obs_size = 6 + default_tracker_num_features + n_actions
    if use_bow_embedder:
        obs_size += word_vocab_size
    if embedder_dim:
        obs_size += embedder_dim
    if callable(intent_classifier):
        obs_size += len(intents)
    # log.info(f"Calculated input size for `GoalOrientedBotNetwork` is {obs_size}")
    return obs_size


def configure_attn(attn,
                   embedder_dim,
                   n_actions,
                   intent_classifier,
                   intents):
    curr_attn_token_size = attn.get('token_size'),
    curr_attn_action_as_key = attn.get('action_as_key'),
    curr_attn_intent_as_key = attn.get('intent_as_key'),
    curr_attn_key_size = attn.get('key_size')

    token_size = curr_attn_token_size or embedder_dim
    action_as_key = curr_attn_action_as_key or False
    intent_as_key = curr_attn_intent_as_key or False

    possible_key_size = 0
    if action_as_key:
        possible_key_size += n_actions
    if intent_as_key and callable(intent_classifier):
        possible_key_size += len(intents)
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


log = getLogger(__name__)


class NNStuffHandler(LRScheduledTFModel):
    SAVE_LOAD_SUBDIR_NAME = "nn_stuff"

    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size", "attention_mechanism"]
    SERIALIZABLE_FIELDS = ["hidden_size", "action_size", "dense_size", "dropout_rate", "l2_reg_coef",
                           "attention_mechanism"]
    UNSUPPORTED = ["obs_size"]
    DEPRECATED = ["end_learning_rate", "decay_steps", "decay_power"]

    def __init__(self,
                 hidden_size,
                 action_size,
                 dropout_rate,
                 l2_reg_coef,
                 dense_size,
                 attention_mechanism,
                 network_parameters,
                 embedder_dim,
                 n_actions,
                 intent_classifier,
                 intents,
                 default_tracker_num_features,
                 use_bow_embedder,
                 word_vocab_size,
                 load_path,
                 save_path,
                 **kwargs):

        if network_parameters is None:
            network_parameters = {}

        if 'learning_rate' in network_parameters:
            kwargs['learning_rate'] = network_parameters.pop('learning_rate')  # todo почему это так, @лёша?

        super().__init__(load_path=load_path, save_path=save_path, **kwargs)

        self.log_deprecated_params(network_parameters.keys())

        self.hidden_size = network_parameters.get("hidden_size", hidden_size)
        self.action_size = network_parameters.get("action_size", action_size) or n_actions
        self.dropout_rate = network_parameters.get("dropout_rate", dropout_rate)
        self.l2_reg_coef = network_parameters.get("l2_reg_coef", l2_reg_coef)
        self.dense_size = network_parameters.get("dense_size", dense_size) or hidden_size

        self.obs_size = calc_obs_size(default_tracker_num_features, n_actions,
                                      use_bow_embedder, word_vocab_size, embedder_dim,
                                      intent_classifier, intents)

        attn = network_parameters.get('attention_mechanism', attention_mechanism)
        if attn:
            self.attention_mechanism = configure_attn(attn, embedder_dim=embedder_dim, n_actions=n_actions,
                                                      intent_classifier=intent_classifier, intents=intents)
            self.obs_size -= self.attention_mechanism.token_size
        else:
            self.attention_mechanism = None

        self._build_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.train_checkpoint_exists():
            # todo переделать
            log.info(f"[initializing `{self.__class__.__name__}` from saved]")
            self.load()
        else:
            log.info(f"[initializing `{self.__class__.__name__}` from scratch]")

    def log_deprecated_params(self, network_parameters):
        if any(p in network_parameters for p in self.DEPRECATED):
            log.warning(f"parameters {self.DEPRECATED} are deprecated,"
                        f" for learning rate schedule documentation see"
                        f" deeppavlov.core.models.lr_scheduled_tf_model"
                        f" or read a github tutorial on super convergence.")

    def _build_graph(self) -> None:
        self._add_placeholders()

        _logits, self._state = self._build_body()

        # probabilities normalization : elemwise multiply with action mask
        _logits_exp = tf.multiply(tf.exp(_logits), self._action_mask)
        _logits_exp_sum = tf.expand_dims(tf.reduce_sum(_logits_exp, -1), -1)
        self._probs = tf.squeeze(_logits_exp / _logits_exp_sum, name='probs')

        # loss, train and predict operations
        self._prediction = tf.argmax(self._probs, axis=-1, name='prediction')

        # _weights = tf.expand_dims(self._utterance_mask, -1)
        # TODO: try multiplying logits to action_mask
        onehots = tf.one_hot(self._action, self.action_size)
        _loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=_logits, labels=onehots
        )
        # multiply with batch utterance mask
        _loss_tensor = tf.multiply(_loss_tensor, self._utterance_mask)
        self._loss = tf.reduce_mean(_loss_tensor, name='loss')
        self._loss += self.l2_reg_coef * tf.losses.get_regularization_loss()
        self._train_op = self.get_train_op(self._loss)

    def _add_placeholders(self) -> None:
        self._dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_prob')

        self._features = tf.placeholder(tf.float32, [None, None, self.obs_size], name='features')

        self._action = tf.placeholder(tf.int32, [None, None], name='ground_truth_action')

        self._action_mask = tf.placeholder(tf.float32, [None, None, self.action_size], name='action_mask')

        self._utterance_mask = tf.placeholder(tf.float32, shape=[None, None], name='utterance_mask')

        self._batch_size = tf.shape(self._features)[0]

        zero_state = tf.zeros([self._batch_size, self.hidden_size], dtype=tf.float32)
        _initial_state_c = tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        _initial_state_h = tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c, _initial_state_h)

        if self.attention_mechanism:
            _emb_context_shape = [None, None, self.attention_mechanism.max_num_tokens, self.attention_mechanism.token_size]
            self._emb_context = tf.placeholder(tf.float32, _emb_context_shape, name='emb_context')
            self._key = tf.placeholder(tf.float32, [None, None, self.attention_mechanism.key_size], name='key')

    def _build_body(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # input projection
        _units = tf.layers.dense(self._features, self.dense_size,
                                 kernel_regularizer=tf.nn.l2_loss, kernel_initializer=xav())

        if self.attention_mechanism:
            _attn_output = self._build_attn_body()
            _units = tf.concat([_units, _attn_output], -1)

        _units = tf_layers.variational_dropout(_units, keep_prob=self._dropout_keep_prob)

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        _utter_lengths = tf.cast(tf.reduce_sum(self._utterance_mask, axis=-1), tf.int32)

        # _output: [batch_size, max_time, hidden_size]
        # _state: tuple of two [batch_size, hidden_size]
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell, _units,
                                            time_major=False, initial_state=self._initial_state,
                                            sequence_length=_utter_lengths)

        _output = tf.reshape(_output, (self._batch_size, -1, self.hidden_size))
        _output = tf_layers.variational_dropout(_output, keep_prob=self._dropout_keep_prob)
        # output projection
        _logits = tf.layers.dense(_output, self.action_size,
                                  kernel_regularizer=tf.nn.l2_loss, kernel_initializer=xav(), name='logits')
        return _logits, _state

    def _build_attn_body(self):
        attn_scope = f"attention_mechanism/{self.attention_mechanism.type}"
        with tf.variable_scope(attn_scope):
            if self.attention_mechanism.type == 'general':
                _attn_output = am.general_attention(self._key, self._emb_context,
                                                    hidden_size=self.attention_mechanism.hidden_size,
                                                    projected_align=self.attention_mechanism.projected_align)
            elif self.attention_mechanism.type == 'bahdanau':
                _attn_output = am.bahdanau_attention(self._key, self._emb_context,
                                                     hidden_size=self.attention_mechanism.hidden_size,
                                                     projected_align=self.attention_mechanism.projected_align)
            elif self.attention_mechanism.type == 'cs_general':
                _attn_output = am.cs_general_attention(self._key, self._emb_context,
                                                       hidden_size=self.attention_mechanism.hidden_size,
                                                       depth=self.attention_mechanism.depth,
                                                       projected_align=self.attention_mechanism.projected_align)
            elif self.attention_mechanism.type == 'cs_bahdanau':
                _attn_output = am.cs_bahdanau_attention(self._key, self._emb_context,
                                                        hidden_size=self.attention_mechanism.hidden_size,
                                                        depth=self.attention_mechanism.depth,
                                                        projected_align=self.attention_mechanism.projected_align)
            elif self.attention_mechanism.type == 'light_general':
                _attn_output = am.light_general_attention(self._key, self._emb_context,
                                                          hidden_size=self.attention_mechanism.hidden_size,
                                                          projected_align=self.attention_mechanism.projected_align)
            elif self.attention_mechanism.type == 'light_bahdanau':
                _attn_output = am.light_bahdanau_attention(self._key, self._emb_context,
                                                           hidden_size=self.attention_mechanism.hidden_size,
                                                           projected_align=self.attention_mechanism.projected_align)
            else:
                raise ValueError("wrong value for attention mechanism type")
        return _attn_output

    def train_checkpoint_exists(self):
        return tf.train.checkpoint_exists(str(self.load_path.resolve()))

    def get_attn_hyperparams(self) -> Optional[GobotAttnHyperParams]:
        attn_hyperparams = None
        if self.attention_mechanism:
            attn_hyperparams = GobotAttnHyperParams(self.attention_mechanism)
        return attn_hyperparams

    def __call__(self, *args, **kwargs):
        # todo _network_call
        pass

    def _network_call(self, utterance_features: UtteranceFeatures,
                      states_c: np.ndarray, states_h: np.ndarray, prob: bool = False) -> Sequence[np.ndarray]:

        dialogue_features = DialogueFeatures()
        dialogue_features.append(utterance_features)
        batch_dialogues_features = BatchDialoguesFeatures(1)
        batch_dialogues_features.append(dialogue_features)

        states_c = [[states_c]]  # list of list aka batch of dialogues
        states_h = [[states_h]]  # list of list aka batch of dialogues

        feed_dict = {
            self._dropout_keep_prob: 1.,
            self._initial_state: (states_c, states_h),
            self._utterance_mask: batch_dialogues_features.b_padded_dialogue_length_mask,
            self._features: batch_dialogues_features.b_featuress,
            self._action_mask: batch_dialogues_features.b_action_masks
        }
        if self.attention_mechanism:
            feed_dict[self._emb_context] = batch_dialogues_features.b_tokens_embeddings_paddeds
            feed_dict[self._key] = batch_dialogues_features.b_attn_keys

        probs, prediction, state = self.sess.run([self._probs, self._prediction, self._state], feed_dict=feed_dict)

        if prob:
            return probs, state[0], state[1]
        return prediction, state[0], state[1]

    def train_on_batch(self, x: list, y: list):
        # todo
        pass

    def _network_train_on_batch(self, batch_features: BatchDialoguesFeatures) -> dict:

        feed_dict = {
            self._dropout_keep_prob: 1.,
            self._utterance_mask: batch_features.b_padded_dialogue_length_mask,
            self._features: batch_features.b_featuress,
            self._action: batch_features.b_action_ids,
            self._action_mask: batch_features.b_action_masks
        }

        if self.attention_mechanism:
            feed_dict[self._emb_context] = batch_features.b_tokens_embeddings_paddeds
            feed_dict[self._key] = batch_features.b_attn_keys

        _, loss_value, prediction = self.sess.run([self._train_op, self._loss, self._prediction], feed_dict=feed_dict)

        return {'loss': loss_value,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum()}

    def load(self, *args, **kwargs) -> None:
        self._load_nn_params()
        super().load(*args, **kwargs)

    def _load_nn_params(self) -> None:
        path = str(self.load_path.with_suffix('.json').resolve())
        # log.info(f"[loading parameters from {path}]")
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.__getattribute__(p) != params.get(p) and p not in {'attn', 'attention_mechanism'}:
                # todo backward-compatible attention serialization
                raise ConfigError(f"`{p}` parameter must be equal to saved"
                                  f" model parameter value `{params.get(p)}`,"
                                  f" but is equal to `{self.__getattribute__(p)}`")

    def save(self, *args, **kwargs) -> None:
        super().save(*args, **kwargs)
        self._save_nn_params()

    def _save_nn_params(self) -> None:
        path = str(self.save_path.with_suffix('.json').resolve())
        nn_params = {opt: self.__getattribute__(opt) for opt in self.SERIALIZABLE_FIELDS}
        # log.info(f"[saving parameters to {path}]")
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(nn_params, fp)
