#coding:utf-8
###################################################
# File Name: covert.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年04月01日 星期一 16时05分26秒
#=============================================================
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Convert Google official BERT models to Fluid parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import joblib
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def define_variable(params, fluid_name='', tf_name=''):
    variable = tf.Variable(tf.convert_to_tensor(params[fluid_name]), name=tf_name)

def np_to_tensor(params):
    tf_prefix = 'bert'

    #embeddings
    tf_embedding_prefix = tf_prefix + '/embeddings'
    define_variable(params, 'pre_encoder_layer_norm_scale', tf_embedding_prefix + '/LayerNorm/gamma')
    define_variable(params, 'pre_encoder_layer_norm_bias',  tf_embedding_prefix + '/LayerNorm/beta')
    define_variable(params, 'pos_embedding',                tf_embedding_prefix + '/position_embeddings')
    define_variable(params, 'word_embedding',               tf_embedding_prefix + '/word_embeddings')
    define_variable(params, 'sent_embedding',               tf_embedding_prefix + '/token_type_embeddings')

    #layers
    tf_encoder_prefix = tf_prefix + '/encoder/layer_'
    for i in range(12):
        fluid_prefix = "encoder_layer_" + str(i)
        define_variable(params, fluid_prefix + '_post_att_layer_norm_scale',     tf_encoder_prefix + str(i) + '/attention/output/LayerNorm/gamma')
        define_variable(params, fluid_prefix + '_post_att_layer_norm_bias',      tf_encoder_prefix + str(i) + '/attention/output/LayerNorm/beta')
        define_variable(params, fluid_prefix + '_multi_head_att_output_fc.w_0',  tf_encoder_prefix + str(i) + '/attention/output/dense/kernel')
        define_variable(params, fluid_prefix + '_multi_head_att_output_fc.b_0',  tf_encoder_prefix + str(i) + '/attention/output/dense/bias')
        define_variable(params, fluid_prefix + '_multi_head_att_key_fc.w_0',     tf_encoder_prefix + str(i) + '/attention/self/key/kernel')
        define_variable(params, fluid_prefix + '_multi_head_att_key_fc.b_0',     tf_encoder_prefix + str(i) + '/attention/self/key/bias')
        define_variable(params, fluid_prefix + '_multi_head_att_query_fc.w_0',   tf_encoder_prefix + str(i) + '/attention/self/query/kernel')
        define_variable(params, fluid_prefix + '_multi_head_att_query_fc.b_0',   tf_encoder_prefix + str(i) + '/attention/self/query/bias')
        define_variable(params, fluid_prefix + '_multi_head_att_value_fc.w_0',   tf_encoder_prefix + str(i) + '/attention/self/value/kernel')
        define_variable(params, fluid_prefix + '_multi_head_att_value_fc.b_0',   tf_encoder_prefix + str(i) + '/attention/self/value/bias')
        define_variable(params, fluid_prefix + '_ffn_fc_0.w_0',                  tf_encoder_prefix + str(i) + '/intermediate/dense/kernel')
        define_variable(params, fluid_prefix + '_ffn_fc_0.b_0',                  tf_encoder_prefix + str(i) + '/intermediate/dense/bias')
        define_variable(params, fluid_prefix + '_post_ffn_layer_norm_scale',     tf_encoder_prefix + str(i) + '/output/LayerNorm/gamma')
        define_variable(params, fluid_prefix + '_post_ffn_layer_norm_bias',      tf_encoder_prefix + str(i) + '/output/LayerNorm/beta')
        define_variable(params, fluid_prefix + '_ffn_fc_1.w_0',                  tf_encoder_prefix + str(i) + '/output/dense/kernel')
        define_variable(params, fluid_prefix + '_ffn_fc_1.b_0',                  tf_encoder_prefix + str(i) + '/output/dense/bias')
    
    #pooler
    tf_pooler_prefix = tf_prefix + '/pooler'
    define_variable(params, 'pooled_fc.w_0',  tf_pooler_prefix + '/dense/kernel')
    define_variable(params, 'pooled_fc.b_0',  tf_pooler_prefix + '/dense/bias')


    #cls
    #define_variable(params, 'mask_lm_out_fc.b_0',               'cls/predictions/output_bias')
    #define_variable(params, 'mask_lm_trans_layer_norm_scale',   'cls/predictions/transform/LayerNorm/gamma')
    #define_variable(params, 'mask_lm_trans_layer_norm_bias',    'cls/predictions/transform/LayerNorm/beta')
    #define_variable(params, 'mask_lm_trans_fc.w_0',             'cls/predictions/transform/dense/kernel')
    #define_variable(params, 'mask_lm_trans_fc.b_0',             'cls/predictions/transform/dense/bias')
    #define_variable(params, 'next_sent_fc.w_0',                 'cls/seq_relationship/output_weights')
    #define_variable(params, 'next_sent_fc.b_0',                 'cls/seq_relationship/output_bias')
    #define_variable(params, 'cls_squad_out_w',                  'cls/squad/output_weights')
    #define_variable(params, 'cls_squad_out_b',                  'cls/squad/output_bias')




def covert(input_file):
    params = joblib.load(input_file)


    graph = tf.Graph()
    with graph.as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        #sess_config = tf.ConfigProto(gpu_options=gpu_options)
        #sess = tf.Session(sess_config)
        sess = tf.Session()
        np_to_tensor(params)


        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        with sess.as_default():
            checkpoint_dir = 'checkpoints'
            checkpoint_prefix = os.path.join(checkpoint_dir, 'bert_model.ckpt')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_prefix)



if __name__ == '__main__':
    covert('params.dict')
    pass
