import configparser
import argparse
import collections
import numpy as np
import tensorflow as tf

import model
import dictionary


class Params(object):
    def __init__(self, d):
        self.__dict__ = d


if __name__ == '__main__':
    # --- load params -------------------------------------------------------------------------------------------------
    # config = configparser.ConfigParser()
    # config.read('default_params.ini')
    # defaults = dict(config['defaults'])
    # params = Params(defaults)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--in', type=str, default="This is a default sentence. Let's see what happens.")
    # args = parser.parse_args()
    # --- load params -------------------------------------------------------------------------------------------------
    config = configparser.ConfigParser()
    config.read('default_params.ini')
    defaults = config['defaults']
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default=defaults['log_dir'])
    parser.add_argument('--save_dir', type=str, default=defaults['save_dir'])
    parser.add_argument('--data_dir', type=str, default=defaults['data_dir'])
    parser.add_argument('--submission_dir', type=str, default=defaults['submission_dir'])
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=defaults['seed'])
    parser.add_argument('--n_iter', type=int, default=defaults['n_iter'])
    parser.add_argument('--n_batch', type=int, default=defaults['n_batch'])
    parser.add_argument('--max_grad_norm', type=int, default=defaults['max_grad_norm'])
    parser.add_argument('--lr', type=float, default=defaults['lr'])
    parser.add_argument('--lr_warmup', type=float, default=defaults['lr_warmup'])
    parser.add_argument('--n_ctx', type=int, default=defaults['n_ctx'])
    parser.add_argument('--n_embd', type=int, default=defaults['n_embd'])
    parser.add_argument('--n_head', type=int, default=defaults['n_head'])
    parser.add_argument('--n_layer', type=int, default=defaults['n_layer'])
    parser.add_argument('--embd_pdrop', type=float, default=defaults['embd_pdrop'])
    parser.add_argument('--attn_pdrop', type=float, default=defaults['attn_pdrop'])
    parser.add_argument('--resid_pdrop', type=float, default=defaults['resid_pdrop'])
    parser.add_argument('--clf_pdrop', type=float, default=defaults['clf_pdrop'])
    parser.add_argument('--l2', type=float, default=defaults['l2'])
    parser.add_argument('--vector_l2', action='store_true', default=defaults['vector_l2'])
    parser.add_argument('--n_gpu', type=int, default=defaults['n_gpu'])
    parser.add_argument('--opt', type=str, default=defaults['opt'])
    parser.add_argument('--afn', type=str, default=defaults['afn'])
    parser.add_argument('--lr_schedule', type=str, default=defaults['lr_schedule'])
    parser.add_argument('--encoder_path', type=str, default=defaults['encoder_path'])
    parser.add_argument('--bpe_path', type=str, default=defaults['bpe_path'])
    parser.add_argument('--n_transfer', type=int, default=defaults['n_transfer'])
    parser.add_argument('--lm_coef', type=float, default=defaults['lm_coef'])
    parser.add_argument('--b1', type=float, default=defaults['b1'])
    parser.add_argument('--b2', type=float, default=defaults['b2'])
    parser.add_argument('--e', type=float, default=defaults['e'])
    params = parser.parse_args()
    print(params)

    # --- prepare example ---------------------------------------------------------------------------------------------
    dict_obj = dictionary.Dictionary(params=params)
    params.n_ctx = 256
    example = "Other critiques of literary theory in narrative challenge the very role of literariness in narrative, as well as the role of narrative in literature. Meaning, narratives and their associated aesthetics, emotions, and values have the ability to operate without the presence of literature and vice versa. According to Didier Costa, the structural model used by Todorov and others is unfairly biased towards a Western interpretation of narrative, and that a more comprehensive and transformative model must be created in order to properly analyze narrative discourse in literature."
    # example = "how are you ?"
    encoded_sample = dict_obj.txt_to_int(example)
    # x, m = dict_obj.transform(encoded_sample)
    x2 = dict_obj.transform_v2(encoded_sample[0], add=1)

    stop = "here"

    # --- generate model as tensorflow graph (train) ------------------------------------------------------------------
    transformer_decoder = model.TransformerDecoder(params=params)
    X = tf.placeholder(tf.int32, [1, None, 2])
    M = tf.placeholder(tf.float32, [1, None])
    Y = tf.placeholder(tf.int32, [None])
    clf_logits, lm_logits, clf_loss, lm_loss = transformer_decoder.model(X,
                                                                         M,
                                                                         Y,
                                                                         train=False,
                                                                         reuse=tf.AUTO_REUSE,
                                                                         greedy_decoding=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # --- load model --------------------------------------------------------------------------------------------------
    transformer_decoder.init_and_load_parameter_from_file(sess=sess, path="model/")

    # --- compute result
    # logits = sess.run(lm_logits, {X: x, M: m})

    # --- perform greedy decoding -------------------------------------------------------------------------------------
    cnt = 0
    while True:
        logits = sess.run(lm_logits, {X: x2})
        values = np.argmax(logits, 1)
        value = values[-1]
        text = x2[0, :-1, 0]
        text = np.append(text, value)
        x2 = dict_obj.transform_v2(text, add=1)
        if value == 267 or cnt > 10:
            break
        cnt += 1
    result = dict_obj.int_to_txt(text)
    stop = "here"


