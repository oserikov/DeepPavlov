from logging import getLogger
from typing import Tuple

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.core.models.component import Component
from deeppavlov.models.embedders.abstract_embedder import Embedder
from deeppavlov.core.layers.tf_layers import cudnn_bi_gru, variational_dropout, INITIALIZER
from deeppavlov.models.squad.utils import CudnnGRU, CudnnCompatibleGRU, dot_attention, softmax_mask


@register('two_sentences_emb')
class TwoSentencesEmbedder(Component):
    def __init__(self, embedder: Embedder, **kwargs):
        self.embedder = embedder

    def __call__(self, sentence_tokens_1, sentence_tokens_2):
        sentence_token_embs_1 = self.embedder(sentence_tokens_1)
        sentence_token_embs_2 = self.embedder(sentence_tokens_2)
        return sentence_token_embs_1, sentence_token_embs_2

@register('rel_ranker')
class RelRanker(LRScheduledTFModel):
    def __init__(self, n_classes: int = 2, dropout_keep_prob: float = 0.5, return_probas: bool = False, **kwargs):
        if 'learning_rate_drop_div' not in kwargs:
            kwargs['learning_rate_drop_div'] = 10.0
        if 'learning_rate_drop_patience' not in kwargs:
            kwargs['learning_rate_drop_patience'] = 5.0
        if 'clip_norm' not in kwargs:
            kwargs['clip_norm'] = 5.0

        super().__init__(**kwargs)
        print("load_path", kwargs)
        print("load_path", self.load_path)

        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.return_probas = return_probas
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        GRU = CudnnGRU

        self.question_ph = tf.placeholder(tf.float32, [None, None, 300])
        self.rel_emb_ph = tf.placeholder(tf.float32, [None, None, 300])
        
        r_mask_2 = tf.cast(self.rel_emb_ph, tf.bool)
        r_len_2 = tf.reduce_sum(tf.cast(r_mask_2, tf.int32), axis=2)
        r_mask = tf.cast(r_len_2, tf.bool)
        r_len = tf.reduce_sum(tf.cast(r_mask, tf.int32), axis=1)
        r_len = tf.expand_dims(r_len, axis=1)
                
        rel_emb = tf.math.divide(tf.reduce_sum(self.rel_emb_ph, axis=1), tf.cast(r_len, tf.float32))

        self.y_ph = tf.placeholder(tf.int32, shape=(None,))
        one_hot_labels = tf.one_hot(self.y_ph, depth=self.n_classes, dtype=tf.float32)
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')

        q_mask_2 = tf.cast(self.question_ph, tf.bool)
        q_len_2 = tf.reduce_sum(tf.cast(q_mask_2, tf.int32), axis=2)
        q_mask = tf.cast(q_len_2, tf.bool)
        q_len = tf.reduce_sum(tf.cast(q_mask, tf.int32), axis=1)

        question_dr = variational_dropout(self.question_ph, keep_prob=self.keep_prob_ph)
        b_size = tf.shape(self.question_ph)[0]

        with tf.variable_scope("context_dependent"):

            rnn = GRU(num_layers=2, num_units=75, batch_size=b_size, input_size=300, keep_prob=self.keep_prob_ph)
            q = rnn(question_dr, seq_len=q_len)

            rel_emb_exp = tf.expand_dims(rel_emb, axis=1)
            dot_products = tf.reduce_sum(tf.multiply(q, rel_emb_exp), axis=2, keep_dims=False)
            s_mask = softmax_mask(dot_products, q_mask)
            att_weights = tf.expand_dims(tf.nn.softmax(s_mask), axis=2)
            s_r = tf.reduce_sum(tf.multiply(att_weights, q), axis=1)

            self.logits = tf.layers.dense(tf.multiply(s_r, rel_emb), 2, activation = None, use_bias = False)
            self.y_pred = tf.argmax(self.logits, axis=-1)

            loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=self.logits)
            
            self.loss = tf.reduce_mean(loss_tensor)
            self.train_op = self.get_train_op(self.loss)

        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())
        self.load()

    def fill_feed_dict(self, xs, y=None, train=False):
        xs = list(xs)
        xs[0] = np.array(xs[0])
        xs[1] = np.array(xs[1])
        feed_dict = {self.question_ph: xs[0], self.rel_emb_ph: xs[1]}
        if y is not None:
            feed_dict[self.y_ph] = y
        if train:
            feed_dict[self.keep_prob_ph] = self.dropout_keep_prob
        if not train:
            feed_dict[self.keep_prob_ph] = 1.0
        
        return feed_dict

    def predict(self, xs):
        feed_dict = self.fill_feed_dict(xs)
        if self.return_probas:
            pred = self.sess.run(self.logits, feed_dict)
        else:
            pred = self.sess.run(self.y_pred, feed_dict)
        return pred

    def __call__(self, *args, **kwargs):
        if len(args[0]) == 0 or (len(args[0]) == 1 and len(args[0][0]) == 0):
            return []
        return self.predict(args)    

    def train_on_batch(self, *args):
        *xs, y = args
        feed_dict = self.fill_feed_dict(xs, y, train=True)
        _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict)
        return {'loss': loss_value,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum()}

    def process_event(self, event_name, data):
        super().process_event(event_name, data)
