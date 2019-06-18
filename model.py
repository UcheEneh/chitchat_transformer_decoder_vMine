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
        x = tf.nn.dropout(x, 1 - pdrop)     # mask out some weight nodes to prevent overfitting
    return x


def embed(X, we):
    we = utils.convert_gradient_to_tensor(we)
    e = tf.gather(we, X)    # get embeddings of X from we (weight of all embeddings)
    # rocstories: we:[(vocab_size + 3 + 77), 768], X:[length_of_input_sequences in 1-D tensors of [77, 2]]; (3: n_special). So e: would give a [?, 77, 2, 768] where ?: length of input batch sequence for the current /gpu:X
    h = tf.reduce_sum(e, 2)     # returns a [?, 77, 768] (i.e performs sum along axis 2: add positional embeddings to the input embedding)
    return h


def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02),  # for train, x: embed_input [? (len of input seq for curr /gpu:X), 77, 768], scope, nf: (n_state (i.e. last element in x shape_list = 768 (for bpe)) * 3), rf: 1,
           b_init=tf.constant_initializer(0), pad='VALID', train=False):        # for mlp, nf: (last element in x shape_list) * 4; rf: still = 1
    with tf.variable_scope(scope):
        nx = utils.shape_list(x)[-1]    # last value in x shape_list
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)      # [1, nx (last element in input x), nf (last element in input x * 3)]   # 768*3 for q, k, v
        b = tf.get_variable("b", [nf], initializer=b_init)  # nf (last element in input x * 3)
        if rf == 1:     # faster 1x1 conv    # Basically, reshape x and w to multiple 1-D tensors, perform dot-product, add bias and the reshape to output format (see next comment)
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, utils.shape_list(x)[:-1]+[nf])    # output format: list of x shape_list except the last value (i.e [?, 77]), concatenated with nf [768 * 3]
        else: # was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):   # after dropout, x: multiple [-1, 768], ny = 1
    with tf.variable_scope('clf'):
        nx = utils.shape_list(x)[-1]    # last value of x shape_list: 768
        w = tf.get_variable("w", [nx, ny], initializer=w_init)  # trainable weight for getting the best token to return for each input token in x
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b      # similarity method to show how related the weight and input are dot product of: [x[0], 768] * [768, 1] + b. The higher values represent more related


def split_heads(x, n, k=False):     # For attention, x: q, k, or v; n: no_of_attention_heads (i.e. 12)
    if k:   # k: True when x = k
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])   # rearrange tensor shape (no of input sequence remains same position, num attention head is now second
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])   # query and value same last two shapes, key is different (Maybe for multiplication purposes during similarity check)


def split_states(x, n):
    x_shape = utils.shape_list(x)
    m = x_shape[-1]     # last value in x shape_list
    new_x_shape = x_shape[:-1]+[n, m//n]    # list of s shape_list except last value concatenated with attention head split (i.e 12 * (m/12) which is still = 12)
    return tf.reshape(x, new_x_shape)


def mask_attn_weights(w):   # w: similarity(q, k)
    n = utils.shape_list(w)[-1]     # shape of values indicating similarity strength
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)     # not sure exactly but it masks the Lower triangular part of the matrix (whatever that means)
    b = tf.reshape(b, [1, 1, n, n])     # reshape the mask for the future words
    w = w*b + -1e9*(1-b)    # set the unmasked (future) words to -inf before softmax step
    return w


def merge_states(x):
    x_shape = utils.shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]  # concatenate list of x_shaoe except the last two values in list, with the multiplication of the last two values
    return tf.reshape(x, new_x_shape)


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))  # returns it to the former shape (i.e. shape it was before it was split into n_heads shape (i guess))


def norm(x, scope, axis=None):  # perform normalization across the input
    if axis is None:
        axis = [-1]

    def _norm(x, g=None, b=None, e=1e-5, axis=None):
        if axis is None:
            axis = [-1]
        u = tf.reduce_mean(x, axis=axis, keep_dims=True)    # mean along last axis ([-1]), but retain the dim. *keep_dims is deprecated, use keepdims instead
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keep_dims=True)     # || x- mean ||**2
        x = (x - u) * tf.rsqrt(s + e)       # e: added to prevent a division by 0
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
        xs = (tf.split(x, self.params.n_gpu, 0) for x in xs)    # split input data into number of gpus (4 for Fab, 2 for me)
        for i, xs in enumerate(zip(*xs)):
            do_reuse = True if i > 0 else None
            with tf.device(utils.assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(),    # variable foo/gpu:X can be shared in a reusing scope, else gives error
                                                                                reuse=do_reuse):
                # logits: the result from the last layer, loss: the difference between this result and label
                clf_logits, lm_logits, clf_losses, lm_losses = self.model(*xs, train=True, reuse=do_reuse)  # assign each input to the model and build train graph
                # clf_logits: [?, 2], clf_loss: [?] where ?: shape of current batch input;  logits is ,2 because we are classifying btwn two diff input seqns
                # for train, these results operation are also used to perform gradient descent and update in gpu (unlike in mgpu_predict where they are just used to only calc the themselves in the gpu)
                if self.params.head_type == "clf":
                    if self.params.lm_coef > 0:     # calculate and apply a joint loss if clf task also includes lm
                        train_loss = tf.reduce_mean(clf_losses) + self.params.lm_coef * tf.reduce_mean(lm_losses)
                    else:
                        train_loss = tf.reduce_mean(clf_losses)
                elif self.params.head_type == "lm":
                    train_loss = tf.reduce_mean(lm_losses)
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))
                tvars = utils.find_trainable_variables("model")
                grads = tf.gradients(train_loss, tvars)     # apply gradient diff. calc (Jacobian) to the trainable variables
                grads = list(zip(grads, tvars))     # zips the gradient descent values and the variables to which they are to be applied on
                gpu_grads.append(grads)             # appends the gradient properties from each gpu
                if self.params.head_type == "clf":
                    gpu_ops.append([clf_logits, clf_losses, lm_losses])     # appends the logit and loss outputs from each gpu if clf
                elif self.params.head_type == "lm":
                    gpu_ops.append([lm_losses])     # appends just the loss outputs from each gpu if lm
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]    # concatenate the loss result from the different gpus
        grads = utils.average_grads(gpu_grads)  # contains [an average of the grads from each gpu, and the corresponding variables]

        # Gradient operations (only in train, not in predict)
        # Accumulate gradient and perform update after a certain treshold
        if self.params.gradient_accumulation:       # False for rocstories      # The threshold condition is defined in the train-loop section in __main__ in train.py?
            tvars = utils.find_trainable_variables("model")
            accum_tvars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvars]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]     # operation to assign 0s into a non-trainable tf.Variable of shape tvars
            accum_ops = [accum_tvars[i].assign_add(grad[0]) for i, grad in enumerate(grads)]    # operation to store the average of the grads from each gpu into a non-trainable tf.Variable of shape tvars
            grads = accum_tvars
        else:
            zero_ops = None
            accum_ops = None
            grads = [g for g, p in grads]       # returns only the gradients, not the variables
        # Perform Optimization  (rocstories:- param.opt: adam)
        train = OPT_FNS[self.params.opt](tvars,
                                         grads,
                                         self.params.lr,
                                         partial(LR_SCHEDULES[self.params.lr_schedule], warmup=self.params.lr_warmup),  # i guess for changing the lr decay value over time (Not sure)
                                         self.params.n_updates_total,
                                         l2=self.params.l2,
                                         max_grad_norm=self.params.max_grad_norm,
                                         vector_l2=self.params.vector_l2,
                                         b1=self.params.b1,
                                         b2=self.params.b2,
                                         e=self.params.e)
        return [train, accum_ops, zero_ops] + ops

    def mgpu_predict(self, *xs):    # Prediction *xs: (X_train, M_train, Y_train) for evaluation but for test: (X_test, M_test, Y_test)
        gpu_ops = []
        xs = (tf.split(x, self.params.n_gpu, 0) for x in xs)    # create tuple of split x along axis 0, into number of available gpu
        for i, xs in enumerate(zip(*xs)):
            with tf.device(utils.assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
                clf_logits, lm_logits, clf_losses, lm_losses = self.model(*xs, train=False, reuse=True)     # train = False for prediction
                if self.params.head_type == "clf":
                    gpu_ops.append([clf_logits, clf_losses, lm_losses])     # just append the loss, no need for grads (grad descent) since we are just evaluating
                elif self.params.head_type == "lm":
                    gpu_ops.append([lm_logits, lm_losses])
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]    # concatenate the loss result from the different gpus
        return ops

    def init_and_load_parameter_from_file(self, sess, path):
        tvars = utils.find_trainable_variables('model')
        with open(os.path.join(path, 'params_shapes.json')) as f:   # loads list of shapes from json file
            shapes = json.load(f)       # [[512, 768], [40478, 768], [1, 768, 2304], ..., ]

        """ np.cumsum:
         a = np.array([[1,2,3], [4,5,6]])
         np.cumsum(a) = array([ 1,  3,  6, 10, 15, 21])
        """
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        # load all the np params to a list
        init_params = [np.load(os.path.join(path, 'params_{}.npy'.format(n))) for n in range(10)]

        """ np.split:
        If an index exceeds the dimension of the array along axis, an empty sub-array is returned correspondingly.
        x = np.arange(5.0, 13.0)
        
        np.split(x, [3, 5, 6, 10])      # split x based on the indices in the list
        >>> [array([ 5.,  6.,  7.]),    # first three: [0:3]
            array([ 8.,  9.]),          # next two: [3:5]
            array([ 10.]),              # next one: [5:6]
            array([ 11.,  12.]),        # next four: [6:10], but only two left in the array so only assigns them
            array([], dtype=float64)]   # last ones: [10:], but nothing left so returns empty array
        """
        # concatenate the params on axis 0, split according to offsets list and remove last sub-array
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        # reshape each split (concatenated) param into the corresponding shape
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        # give embeddings to the special params ( _classify_, _delimiter_, ...)
        embeddings_special = (np.random.randn(self.params.n_special, self.params.n_embd)*0.02).astype(np.float32)
        # concat the dictionary embeddings, special embeddings and the learned / pre-trained  input sequence embeddings
        init_embeddings = np.concatenate([init_params[1], embeddings_special, init_params[0][:self.params.n_ctx]])
        init_params[0] = init_embeddings
        del init_params[1]  # delete the vocab / dictionary embeddings
        if self.params.n_transfer == -1:
            self.params.n_transfer = 0
        else:
            # 1 (for the dictionary, special, input emb) + 144 (for the other 12 layers which happen to have 12 trainable variables each :D)
            self.params.n_transfer = 1 + self.params.n_transfer * 12
        sess.run(tf.global_variables_initializer())
        """
        Perform transfer learning: set the first n_transfer variables (see how init_params and tvars looks like)
        tvars contains:
            index 0: weight embeddings (model/we:0) [40558, 768]:- concat of dictionary, special and input seq embeddings,
            index 1: (model/h0/attn/c_attn/w:0) [1, 768, 2304]:- weight for the self attn for similarity calc in layer 0,
            index2: (model/h0/attn/c_attn/b:0) [2304]:- bias for the self attn in layer 0,
            index3: (model/h0/attn/c_proj/w:0) [2304]:- weight for the final attn output after softmax of all similarity output in layer 0,
            ... (c_proj/b:0, layer_norm1, mlp, layer_norm2), ... and so on for the remaining 11 layers

        (The classification weight and bias are not assigned since learning hasn't been done yet for it (so just initialized)
        """
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
        with tf.variable_scope('model', reuse=reuse):   # initially reuse is None (first gpu so doesn't need a reuse), but later reuse = True, so the variable mode/... is shared
            # --- generate embedding matrix ---------------------------------------------------------------------------
            we = tf.get_variable(name="we",
                                 shape=[self.params.n_vocab + self.params.n_special + self.params.n_ctx,    # [(vocab_size + 3 + 77), 768]   # n_ctx: max_seq_length (rocstories = 77, movie = 511)
                                        self.params.n_embd],                                                # n_embd (768): length of the embedding vector (i.e. e.g. GloVe d=50, but this time we use bpe instead)
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            we = dropout(we, self.params.embd_pdrop, train)     # set the dropout for each gpu input

            # --- reshape, if not greedy decoding ---------------------------------------------------------------------
            # Not fully implemented.
            if not greedy_decoding:
                X = tf.reshape(X, [-1, self.params.n_ctx, self.params.n_embd_d+1])      # pass '[-1]' to flatten 'X' to several (no of input sequences) 1-D tensors of [77, 2] for rocstories
                M = tf.reshape(M, [-1, self.params.n_ctx])

            # --- add positional embedding and embed training data ----------------------------------------------------
            h = embed(X, we)    # h: embedded input added with positional encoding all embeddings got from the bpe: we

            # --- decoder stacks --------------------------------------------------------------------------------------
            for layer in range(self.params.n_layer):    # n_layer = 12, maybe change for my own training
                h = self.block(h, 'h%d' % layer, train=train, scale=True)   # returns the final output matrix which contains the result after passing through all the n_layers, mlp, and layer_norm for the input sequence in the current /gpu:X
            # h shape: [?, 77, 768] :- this is the final layer_norm output to be used for multi-task leaarning (e.g. passed to the softmax and cross_entropy_loss)
            # --- language modeling loss ------------------------------------------------------------------------------
            if lm_logits_only:
                lm_h = tf.reshape(h, [-1, self.params.n_embd])      # reshape decoder output to be multiples of n_embd (768): length of the embedding vector (i.e. e.g. GloVe d=50, but this time we use bpe instead)
                lm_logits = tf.nn.softmax(tf.matmul(lm_h, we, transpose_b=True))    # softmax for predicting y
            else:
                lm_h = tf.reshape(h[:, :-1], [-1, self.params.n_embd])  # :-1 - all except the last shape value which is then replaced by the roll out (-1). So: [(? * 77), 768]
                lm_logits = tf.matmul(lm_h, we, transpose_b=True)   # then perform matmul to get most-similar to the full dictionary tokens as the logits [(? * 77), 40558]

            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits,    # why is the first row in the middle from all X removed (even in M below) ? Maybe its the start token?
                                                                       labels=tf.reshape(X[:, 1:, 0], [-1]))    # for language modeling, the label y would be the input X (reshape the label to be a single vector) but only one of the inputs (The second input (i.e x13)) was used for the modeling since we aren't performing any classification. The label vector contains the start encoding value for both the input token and the positional encoding for all the input sequences (i.e [1497, :1, :0,  2]) (So there would be a (1497 * 2) label vector
            lm_losses = tf.reshape(lm_losses, [utils.shape_list(X)[0], utils.shape_list(X)[1] - 1])
            lm_losses = (tf.reduce_sum(lm_losses * M[:, 1:], 1) + 1e-9) / (tf.reduce_sum(M[:, 1:], 1) + 1e-9)   # objective fn to maximize: sum of the losses but now using the mask to remove unnecessary positional values from the weight matrix output

            # --- classifier loss -------------------------------------------------------------------------------------
            if self.params.head_type == "clf":
                # logits: the result from the last layer, loss: the difference between this result and label
                clf_logits, clf_losses = self.classifier_head(X=X, Y=Y, h=h, train=train)   # for rocstories, Y = the label for the right answer (0 or 1 indicating which of the 2 in [1497, 2] from X is right)
            elif self.params.head_type == "lm":
                clf_logits, clf_losses = [], []
            else:
                raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))

            return clf_logits, lm_logits, clf_losses, lm_losses

    def classifier_head(self, X, Y, h, train=False):    # rocstories: X: (?, 2, 77, 2), Y:(?,)  h: (?, 77, 768)  # in X, the first two reps the two combinations of the input sentences and the right and wrong answers    # [77, 2], in the 2 here, the second column is just the positional encoding for all input tokens (i.e the input sentences)
        clf_h = tf.reshape(h, [-1, self.params.n_embd])     # reshape decoder output to be multiples of n_embd (768). So: clf_h: [(? * 77), 768]
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], self.params.clf_token), tf.float32), 1), tf.int32)    # tf.equal is an element-wise operator so it returns True if any of the elements from X matches the clf_token and false otherwise. The position of the true is then returned as the output.
        clf_h = tf.gather(clf_h, tf.range(utils.shape_list(X)[0], dtype=tf.int32) * self.params.n_ctx + pool_idx)   # make a [? * 77] list vector (of size 768) + index of clf_token (should also be ?)
                                                                                                                    # tf.gather gets a the outputs from h that would be used to perform the classification and backpropagation based on the indices tf.range... (So we have a list of the values in h containing the first (? * 77) tokens and also the index of the clf or lm tokens
        clf_h = tf.reshape(clf_h, [-1, self.params.clf_pipes, self.params.n_embd])  # reshapes into [?, 2, 768]   # clf_pipes: 2 for (0 or 1)
        if train and self.params.clf_pdrop > 0:     # apply dropout if train otherwise not
            shape = utils.shape_list(clf_h)     # shape: A 1-D Tensor, representing the shape for randomly generated keep/drop flags.
            shape[1] = 1    # By default, each element is kept or dropped independently, but since the shape is specified for row = 1, each row will be kept or not kept together
            clf_h = tf.nn.dropout(clf_h, 1 - self.params.clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, self.params.n_embd])     # after dropout, reshapes into [?, 768]: (?: len_batch_input * 77)
        clf_logits = clf(clf_h, 1, train=train)     # returns the result of the flattened fc layer between the clf_h as input and the weight vector which are back propagatable (trainable)
        clf_logits = tf.reshape(clf_logits, [-1, self.params.clf_pipes])    # reshape to [-1, 2]    # clf_pipes: 2 for (0 or 1) containing the probability of the first or second input sequence being the right one or not

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)    # performs sotmax and then calculates loss between logits and label, Y

        return clf_logits, clf_losses   # logits: the result from the last layer, loss: the difference between this result and label


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


    def block(self, x, scope, train=False, scale=False):    # scope: hNum where Num: current decoder layer, x: embedded input with addition of positional encoding
        with tf.variable_scope(scope):
            nx = utils.shape_list(x)[-1]    # nx: 768 for bpe. This gets last element in shape list. (i guess shape_list is done to deal with 'None' in the shape of x, since a tensor op (tf.shape()) is returned instead of None)
            a = self.attn(x, 'attn', nx, self.params.n_head, train=train, scale=scale)      # n_head: 12 multi-head attention   # returns matmul output of the [softmax(q,k), v] for the attention matrix: [?, 77, 768] which contains the attn result for each input on itself (q = k)
            n = norm(x + a, 'ln_1')     # layer normalization with residual addition (for pos embedding rettention) * Both are of shape [?, 77, 768]
            m = self.mlp(n, 'mlp', nx * 4, train=train)     # multi-layer perceptron (Feed-forward) (Not sure why 4; maybe for depth?)
            h = norm(n + m, 'ln_2')
            return h


    # Multi-layer perceptron
    def mlp(self, x, scope, n_state, train=False):
        with tf.variable_scope(scope):
            nx = utils.shape_list(x)[-1]    # nx: 768 for bpe
            act = ACT_FNS[self.params.afn]  # gelu
            h = act(conv1d(x, 'c_fc', n_state, 1, train=train)) # apply linear transformation on the returned activations
            h2 = conv1d(h, 'c_proj', nx, 1, train=train)        # perform conv (matmul) using a trainable (back-propagatable) weight on the 4 layer fc output which was calc. on the normalized layer output of the attn values
            h2 = dropout(h2, self.params.resid_pdrop, train)
            return h2

    def attn(self, x, scope, n_state, n_head, train=False, scale=False):    # scope: hNum where Num: current decoder layer, x: embedded input with addition of positional encoding
        assert n_state % n_head == 0    # n_head: attention head; n_state: last element in x shape_list.

        def _attn(q, k, v, train=False, scale=False):
            w = tf.matmul(q, k)     # similarity measure as an inner product. # [:, 12, x, y] * [:, 12, y, x] would give us a scalar value at the las two positions indicating the similatity strength [x: 77, y:64] so final = [?, 12, 77, 77]
            if scale:   # scale: divide by sqrt(dim) to make value not too large
                n_state = utils.shape_list(v)[-1]   # last value in v shape_list
                w = w * tf.rsqrt(tf.cast(n_state, tf.float32))
            w = mask_attn_weights(w)
            w = tf.nn.softmax(w)    # still same shape but now contains prob. dist. of the keys that are most similar to the query [?, 12, 77, 77]
            w = dropout(w, self.params.attn_pdrop, train)
            a = tf.matmul(w, v)     # use the distribution to pick the most similar value   # shape back to attn_heads: [?, 12, 77, 64]
            return a

        with tf.variable_scope(scope):      # x: embed_input, scope, n_state: last element in x shape_list * 3, 1, train
            c = conv1d(x, 'c_attn', n_state * 3, 1, train=train)    # make trainable weights for q, k, v (i.e. *3) wrt input x (Note: for self-attention, q,k,v are from the same input)
            q, k, v = tf.split(c, 3, 2)     # split into 3 along axis 2     # split from 2304 (i.e [77, 768 * 3]) to three tensors each of [77, 768] representing weights for the q, k, v
            q = split_heads(q, n_head)      # each input sequence is now split into 12 attention heads where each attention head deals with 64 input embeddings for an embed_size of 768 (64 * 12 = 768)
            k = split_heads(k, n_head, k=True)  # diff. last two shapes from q and v which are both similar     # k: [?, 12, 64, 77]
            v = split_heads(v, n_head)      # q, v: [?, 12, 77, 64]
            a = _attn(q, k, v, train=train, scale=scale)    # returns the matrix but with the most similar value (i.e. after multipyling the softmax distribution with value, v)
            a = merge_heads(a)  # returns it to the former shape (i.e. shape it was before it was split into attn: n_heads shape). So now, shape: [?, 77, 768]
            a = conv1d(a, 'c_proj', n_state, 1, train=train)    # make trainable weights for the attention, matmul with those weights and add bias. return the output
            a = dropout(a, self.params.resid_pdrop, train)
            return a
