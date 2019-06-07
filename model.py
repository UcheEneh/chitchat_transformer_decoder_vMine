import os
import json
import math
import joblib
import tensorflow as tf
import numpy as np
import utils
import opt

from functools import partial
# from tensor2tensor.utils import beam_search


# --- definitions -----------------------------------------------------------------------------------------------------
def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))


def swish(x):
    return x*tf.nn.sigmoid(x)


ACT_FNS = {
    'relu': tf.nn.relu,
    'swish': swish,
    'gelu': gelu
}

LR_SCHEDULES = {
    'warmup_cosine': opt.warmup_cosine,
    'warmup_linear': opt.warmup_linear,
    'warmup_constant': opt.warmup_constant,
}


OPT_FNS = {
    'adam': opt.adam,
}


# --- util functions --------------------------------------------------------------------------------------------------
def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1 - pdrop)
    return x


def embed(X, we):
    we = utils.convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h


def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02),
           b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = utils.shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, utils.shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = utils.shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b


def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def split_states(x, n):
    x_shape = utils.shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)


def mask_attn_weights(w):
    n = utils.shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w


def merge_states(x):
    x_shape = utils.shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def norm(x, scope, axis=None):
    if axis is None:
        axis = [-1]

    def _norm(x, g=None, b=None, e=1e-5, axis=None):
        if axis is None:
            axis = [-1]
        u = tf.reduce_mean(x, axis=axis, keep_dims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keep_dims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x * g + b
        return x

    with tf.variable_scope(scope):
        n_state = utils.shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(x, g, b, axis=axis)


class TransformerDecoder:
    def __init__(self, params):
        self.params = params

    def mgpu_train(self, *xs):
        gpu_ops = []
        gpu_grads = []
        tvars = None
        xs = (tf.split(x, self.params.n_gpu, 0) for x in xs)
        for i, xs in enumerate(zip(*xs)):
            do_reuse = True if i > 0 else None
            with tf.device(utils.assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(),
                                                                                reuse=do_reuse):
                clf_logits, lm_logits, clf_losses, lm_losses = self.model(*xs, train=True, reuse=do_reuse)
                if self.params.head_type == "clf":
                    if self.params.lm_coef > 0:
                        train_loss = tf.reduce_mean(clf_losses) + self.params.lm_coef * tf.reduce_mean(lm_losses)
                    else:
                        train_loss = tf.reduce_mean(clf_losses)
                elif self.params.head_type == "lm":
                    train_loss = tf.reduce_mean(lm_losses)
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))
                tvars = utils.find_trainable_variables("model")
                grads = tf.gradients(train_loss, tvars)
                grads = list(zip(grads, tvars))
                gpu_grads.append(grads)
                if self.params.head_type == "clf":
                    gpu_ops.append([clf_logits, clf_losses, lm_losses])
                elif self.params.head_type == "lm":
                    gpu_ops.append([lm_losses])
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        grads = utils.average_grads(gpu_grads)

        if self.params.gradient_accumulation:
            tvars = utils.find_trainable_variables("model")
            accum_tvars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvars]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]
            accum_ops = [accum_tvars[i].assign_add(grad[0]) for i, grad in enumerate(grads)]
            grads = accum_tvars
        else:
            zero_ops = None
            accum_ops = None
            grads = [g for g, p in grads]

        train = OPT_FNS[self.params.opt](tvars,
                                         grads,
                                         self.params.lr,
                                         partial(LR_SCHEDULES[self.params.lr_schedule], warmup=self.params.lr_warmup),
                                         self.params.n_updates_total,
                                         l2=self.params.l2,
                                         max_grad_norm=self.params.max_grad_norm,
                                         vector_l2=self.params.vector_l2,
                                         b1=self.params.b1,
                                         b2=self.params.b2,
                                         e=self.params.e)
        return [train, accum_ops, zero_ops] + ops

    def mgpu_predict(self, *xs):
        gpu_ops = []
        xs = (tf.split(x, self.params.n_gpu, 0) for x in xs)
        for i, xs in enumerate(zip(*xs)):
            with tf.device(utils.assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
                clf_logits, lm_logits, clf_losses, lm_losses = self.model(*xs, train=False, reuse=True)
                if self.params.head_type == "clf":
                    gpu_ops.append([clf_logits, clf_losses, lm_losses])
                elif self.params.head_type == "lm":
                    gpu_ops.append([lm_logits, lm_losses])
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        return ops

    def init_and_load_parameter_from_file(self, sess, path):
        tvars = utils.find_trainable_variables('model')
        with open(os.path.join(path, 'params_shapes.json')) as f:
            shapes = json.load(f)
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load(os.path.join(path, 'params_{}.npy'.format(n))) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        embeddings_special = (np.random.randn(self.params.n_special, self.params.n_embd)*0.02).astype(np.float32)
        init_embeddings = np.concatenate([init_params[1], embeddings_special, init_params[0][:self.params.n_ctx]])
        init_params[0] = init_embeddings
        del init_params[1]
        if self.params.n_transfer == -1:
            self.params.n_transfer = 0
        else:
            self.params.n_transfer = 1 + self.params.n_transfer * 12
        sess.run(tf.global_variables_initializer())
        sess.run([p.assign(ip) for p, ip in zip(tvars[:self.params.n_transfer], init_params[:self.params.n_transfer])])

    def load_checkpoint(self, sess, path=None):
        if path is None:
            save_dir = os.path.join(self.params.save_dir, self.params.desc, 'best_params.jl')
        else:
            save_dir = path
        t_vars = utils.find_trainable_variables('model')
        sess.run([p.assign(ip) for p, ip in zip(t_vars, joblib.load(os.path.join(save_dir)))])

    def model(self, X, M, Y, train=False, reuse=False, greedy_decoding=False, lm_logits_only=False):
        """ Forward Step of the Decoder.

        To be done:
            - Adapt to multi-positional embeddings (DONE)
            - Adapt to additional tasks beside "classifier"

        Args:
            X       A Tensor. Training data with shape [batch_size, 2, max_len, clf_pipes]
            M       A Tensor. Masks with shape [batch_size, 1, max_len, clf_pipes]
            Y       A Tensor. Classifier labels with shape [batch_size]
        """
        with tf.variable_scope('model', reuse=reuse):
            # --- generate embedding matrix ---------------------------------------------------------------------------
            we = tf.get_variable(name="we",
                                 shape=[self.params.n_vocab + self.params.n_special + self.params.n_ctx,
                                        self.params.n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            we = dropout(we, self.params.embd_pdrop, train)

            # --- reshape, if not greedy decoding ---------------------------------------------------------------------
            # Not fully implemented.
            if not greedy_decoding:
                X = tf.reshape(X, [-1, self.params.n_ctx, self.params.n_embd_d+1])
                M = tf.reshape(M, [-1, self.params.n_ctx])

            # --- add positional embedding and embed training data ----------------------------------------------------
            h = embed(X, we)

            # --- decoder stacks --------------------------------------------------------------------------------------
            for layer in range(self.params.n_layer):
                h = self.block(h, 'h%d' % layer, train=train, scale=True)

            # --- language modeling loss ------------------------------------------------------------------------------
            if lm_logits_only:
                lm_h = tf.reshape(h, [-1, self.params.n_embd])
                lm_logits = tf.nn.softmax(tf.matmul(lm_h, we, transpose_b=True))
            else:
                lm_h = tf.reshape(h[:, :-1], [-1, self.params.n_embd])
                lm_logits = tf.matmul(lm_h, we, transpose_b=True)

            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits,
                                                                       labels=tf.reshape(X[:, 1:, 0], [-1]))
            lm_losses = tf.reshape(lm_losses, [utils.shape_list(X)[0], utils.shape_list(X)[1] - 1])
            lm_losses = (tf.reduce_sum(lm_losses * M[:, 1:], 1) + 1e-9) / (tf.reduce_sum(M[:, 1:], 1) + 1e-9)

            # --- classifier loss -------------------------------------------------------------------------------------
            if self.params.head_type == "clf":
                clf_logits, clf_losses = self.classifier_head(X=X, Y=Y, h=h, train=train)
            elif self.params.head_type == "lm":
                clf_logits, clf_losses = [], []
            else:
                raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))

            return clf_logits, lm_logits, clf_losses, lm_losses

    def classifier_head(self, X, Y, h, train=False):
        clf_h = tf.reshape(h, [-1, self.params.n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], self.params.clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(utils.shape_list(X)[0], dtype=tf.int32) * self.params.n_ctx + pool_idx)

        clf_h = tf.reshape(clf_h, [-1, self.params.clf_pipes, self.params.n_embd])
        if train and self.params.clf_pdrop > 0:
            shape = utils.shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1 - self.params.clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, self.params.n_embd])
        clf_logits = clf(clf_h, 1, train=train)
        clf_logits = tf.reshape(clf_logits, [-1, self.params.clf_pipes])

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)

        return clf_logits, clf_losses


    # def beamforming(self, X, reuse=False):
    #     """ Performs beamforming.
    #     """
    #     with tf.variable_scope('model', reuse=reuse):
    #         we = tf.get_variable(name="we",
    #                              shape=[self.params.n_vocab + self.params.n_special + self.params.n_ctx,
    #                                     self.params.n_embd],
    #                              initializer=tf.random_normal_initializer(stddev=0.02))
    #         h = embed(X, we)
    #         initial_id = X[-1, 0]
    #
    #         def decoder_bf(x):
    #
    #             return x
    #
    #         results, scores = beam_search.beam_search(symbols_to_logits_fn=decoder_bf,
    #                                                   initial_ids=[0],
    #                                                   beam_size=4,
    #                                                   decode_length=16,
    #                                                   vocab_size=self.params.n_vocab,
    #                                                   alpha=0.65,
    #                                                   eos_id=3)  # TODO: Machn Punkt draus.
    #
    #         return results, scores


    def block(self, x, scope, train=False, scale=False):
        with tf.variable_scope(scope):
            nx = utils.shape_list(x)[-1]
            a = self.attn(x, 'attn', nx, self.params.n_head, train=train, scale=scale)
            n = norm(x + a, 'ln_1')
            m = self.mlp(n, 'mlp', nx * 4, train=train)
            h = norm(n + m, 'ln_2')
            return h



    def mlp(self, x, scope, n_state, train=False):
        with tf.variable_scope(scope):
            nx = utils.shape_list(x)[-1]
            act = ACT_FNS[self.params.afn]
            h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
            h2 = conv1d(h, 'c_proj', nx, 1, train=train)
            h2 = dropout(h2, self.params.resid_pdrop, train)
            return h2

    def attn(self, x, scope, n_state, n_head, train=False, scale=False):
        assert n_state % n_head == 0

        def _attn(q, k, v, train=False, scale=False):
            w = tf.matmul(q, k)
            if scale:
                n_state = utils.shape_list(v)[-1]
                w = w * tf.rsqrt(tf.cast(n_state, tf.float32))
            w = mask_attn_weights(w)
            w = tf.nn.softmax(w)
            w = dropout(w, self.params.attn_pdrop, train)
            a = tf.matmul(w, v)
            return a

        with tf.variable_scope(scope):
            c = conv1d(x, 'c_attn', n_state * 3, 1, train=train)
            q, k, v = tf.split(c, 3, 2)
            q = split_heads(q, n_head)
            k = split_heads(k, n_head, k=True)
            v = split_heads(v, n_head)
            a = _attn(q, k, v, train=train, scale=scale)
            a = merge_heads(a)
            a = conv1d(a, 'c_proj', n_state, 1, train=train)
            a = dropout(a, self.params.resid_pdrop, train)
            return a
