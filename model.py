import os
import json
import math
import joblib
import tensorflow as tf
import numpy as np
import utils
import opt

import datetime

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
    """
    For rocstories: - we:[(vocab_size + 3 + 77), 768],
                    - X:[length_of_input_sequences in 1-D tensors of [77, 2]]; (3: n_special).
    e = tf.gather(): - get embeddings of X from we (weight of all embeddings)
                     - So e: would give a [?, 77, 2, 768] where ?: length of input batch sequence for the current /gpu:X
    h = tf.reduce_sum(): returns a [?, 77, 768] (i.e performs sum along axis 2: add pos. embeds to the input embed)
    """
    we = utils.convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h


def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02),
           b_init=tf.constant_initializer(0), pad='VALID', train=False):
    """
    for train, x: embed_input [? (len of input seq for curr /gpu:X), 77, 768], scope, nf: (n_state (i.e. last element in
                    x shape_list = 768 (for bpe)) * 3), rf: 1,
    for mlp, nf: (last element in x shape_list) * 4; rf: still = 1
    w: [1, nx (last element in input x), nf (last element in input x * 3)]   # 768*3 for q, k, v
    b: nf = (last element in input x * 3)
    if rf==1: Basically, reshape x and w to multiple 1-D tensors, perform dot-product, add bias and the reshape to
                output format (see next comment)
    output format: list of x shape_list except the last value (i.e [?, 77]), concatenated with nf [768 * 3]
    """
    with tf.variable_scope(scope):
        nx = utils.shape_list(x)[-1]    # last value in x shape_list
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1:     # faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, utils.shape_list(x)[:-1]+[nf])
        else:   # was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c


def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    """
    after dropout, x: multiple [-1, 768], ny = 1
    x: concatenation of the output prediction (from block i.e. h) of x12 and x13; so the output would be a (?, 1) vector
        for the training of the output prediction
    w: trainable weight for getting the best token to return for each input token in x
    tf.matmul(): similarity method to show how related the weight and input are.
                Dot product of: [x[0], 768] * [768, 1] + b. The higher values represent more related
    """
    with tf.variable_scope('clf'):
        nx = utils.shape_list(x)[-1]    # last value of x shape_list: 768
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b


def split_heads(x, n, k=False):
    """
    For attention, - x: q, k, or v;
                   - n: no_of_attention_heads (i.e. 12)
    if k: rearrange tensor shape (no of input sequence remains same position, num attention head is now second
    else: query and value same last two shapes, key is different (For multiplication purposes during similarity check)
    """
    if k:   # k: True when x = k
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def split_states(x, n):
    x_shape = utils.shape_list(x)
    m = x_shape[-1]     # last value in x shape_list
    # new_x_shape: split x shape_list except last value concat with attn head split (i.e 12*(m/12) which is still = 12)
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)


def mask_attn_weights(w):
    """
    w: similarity(q, k)
    n: shape of values indicating similarity strength
    tf.matrix_band_part(): - leave everything below center diagonal (0) but make everything above center diag. to 0
                           - i.e. masks the Lower triangular part of the matrix, shape: [511, 511]
    tf.reshape: reshape the mask to same rank as w, for the future words
    """
    n = utils.shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)    # set the unmasked (future) words to -inf before softmax step
    return w


def merge_states(x):
    x_shape = utils.shape_list(x)
    # concatenate list of x_shape except the last two values in list, with the multiplication of the last two values
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


# returns x to the former shape (i.e. shape it was before it was split into n_heads shape (i guess))
def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def norm(x, scope, axis=None):  # perform normalization across the input
    if axis is None:
        axis = [-1]

    def _norm(x, g=None, b=None, e=1e-5, axis=None):
        if axis is None:
            axis = [-1]
        u = tf.reduce_mean(x, axis=axis, keep_dims=True)    # mean along last axis ([-1]), but retain the dim.
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


class Transformer:
    def __init__(self, params, use_encoder):
        self.params = params
        self.use_encoder = use_encoder  # Always False if not explicitly set at the call line
        # self.logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + "/")

    def mgpu_train(self, *xs):
        gpu_ops = []
        gpu_grads = []
        tvars = None
        # split input data into number of gpus (4 for Fab, 2 for me, 1 on the computer)
        xs = (tf.split(x, self.params.n_gpu, 0) for x in xs)
        for i, xs in enumerate(zip(*xs)):
            do_reuse = True if i > 0 else None
            """
            reuse: variable foo/gpu:X can be shared in a reusing scope, else gives error
            logits: the result from the last layer, loss: the difference between this result and label
            model(): - assign each input to the model and build train graph
                     - clf_logits: [?, 2], clf_loss: [?] 
                        where ?: shape of current batch input;  
                        logits is [,2] because we are classifying btwn two diff input seqns
            for train: these results operation are also used to perform gradient descent and update in gpu (unlike in 
                        mgpu_predict where they are just used to only calc the themselves in the gpu)
            tf.gradients(): apply gradient diff. calc (Jacobian) to the trainable variables
            grads = list(): zips the gradient descent values and the variables to which they are to be applied on
            gpu_ops.append: appends the logit and loss outputs from each gpu if clf
            """
            with tf.device(utils.assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(),
                                                                                reuse=do_reuse):
                clf_logits, lm_logits, clf_losses, lm_losses = self.model(*xs, train=True, reuse=do_reuse)
                if self.params.head_type == "clf":
                    if self.params.lm_coef > 0:     # calculate and apply a joint loss if clf task also includes lm
                        train_loss = tf.reduce_mean(clf_losses) + self.params.lm_coef * tf.reduce_mean(lm_losses)
                        # tf.summary.scalar('Multi-task Clf-Lm Loss average', train_loss)
                    else:
                        train_loss = tf.reduce_mean(clf_losses)
                        # tf.summary.scalar('Clf Loss average', train_loss)
                elif self.params.head_type == "lm":
                    train_loss = tf.reduce_mean(lm_losses)
                    # tf.summary.scalar('Lm Loss average', train_loss)
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))
                tvars = utils.find_trainable_variables("model")
                grads = tf.gradients(train_loss, tvars)
                grads = list(zip(grads, tvars))
                gpu_grads.append(grads)             # appends the gradient properties from each gpu
                if self.params.head_type == "clf":
                    gpu_ops.append([clf_logits, clf_losses, lm_losses])
                elif self.params.head_type == "lm":
                    gpu_ops.append([lm_losses])     # appends just the loss outputs from each gpu if lm
                else:
                    raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))

        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]    # concatenate the loss result from the different gpus
        # contains [an average of the grads from each gpu, and the corresponding variables]
        grads = utils.average_grads(gpu_grads)

        """
        Gradient operations (only in train, not in predict)
        Accumulate gradient and perform update after a certain treshold. False for rocstories
        The threshold condition is defined in the train-loop section in __main__ in train.py
        
        zero_ops: operation to assign 0s into a non-trainable tf.Variable of shape tvars
        accum_ops: operation to store the average of the grads from each gpu into a non-trainable tf.Variable of shape tvars
        
        else loop: returns only the gradients, not the variables
        """
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

        # Perform Optimization  (rocstories:- param.opt: adam)
        # partial(LR_SCHEDULES...): i guess for changing the lr decay value over time (Not sure)
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

        # Tensorboard
        # self.merged = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter(self.logdir, tf.Session().graph)  # sess.graph
        return [train, accum_ops, zero_ops] + ops

    def mgpu_predict(self, *xs):
        """
        Prediction *xs: (X_train, M_train, Y_train) for evaluation but for test: (X_test, M_test, Y_test)
        tf.split(): create tuple of split x along axis 0, into number of available gpu
        train = False for prediction
        gpu.ops.append(): just append the loss, no need for grads (grad descent) since we are just evaluating
        """
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
        ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]    # concatenate the loss result from the different gpus
        return ops

    def init_and_load_parameter_from_file(self, sess, path):
        tvars = utils.find_trainable_variables('model')
        with open(os.path.join(path, 'params_shapes.json')) as f:   # loads list of shapes from json file
            shapes = json.load(f)       # [[512, 768], [40478, 768], [1, 768, 2304], ..., ]

        """ 
        - np.cumsum:
             a = np.array([[1,2,3], [4,5,6]])
             np.cumsum(a) = array([ 1,  3,  6, 10, 15, 21])
        - load all the np params to a list
        - concatenate the params on axis 0, split according to offsets list and remove last sub-array
            np.split:
            If an index exceeds the dimension of the array along axis, an empty sub-array is returned correspondingly.
            x = np.arange(5.0, 13.0)
            np.split(x, [3, 5, 6, 10])      # split x based on the indices in the list
            >>> [array([ 5.,  6.,  7.]),    # first three: [0:3]
                array([ 8.,  9.]),          # next two: [3:5]
                array([ 10.]),              # next one: [5:6]
                array([ 11.,  12.]),        # next four: [6:10], but only two left in the array so only assigns them
                array([], dtype=float64)]   # last ones: [10:], but nothing left so returns empty array
        - reshape each split (concatenated) param into the corresponding shape
        - give embeddings to the special params ( _classify_, _delimiter_, ...)
        - concat the dictionary embeddings, special embeddings and the learned / pre-trained  input sequence embeddings
        """
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load(os.path.join(path, 'params_{}.npy'.format(n))) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        embeddings_special = (np.random.randn(self.params.n_special, self.params.n_embd)*0.02).astype(np.float32)
        init_embeddings = np.concatenate([init_params[1], embeddings_special, init_params[0][:self.params.n_ctx]])
        init_params[0] = init_embeddings
        del init_params[1]  # delete the vocab / dictionary embeddings
        if self.params.n_transfer == -1:
            self.params.n_transfer = 0
        else:
            # 1 (for we: i.e. the dictionary, special, input emb) +
            # 144 (for the other 12 layers which happen to have 12 trainable variables each :D)
            self.params.n_transfer = 1 + self.params.n_transfer * 12
        sess.run(tf.global_variables_initializer())

        """
        Perform transfer learning: set the first n_transfer variables (see how init_params and tvars looks like)
        tvars contains:
            index 0: weight embeddings (model/we:0) [40558, 768]: concat of dictionary, special and input seq embedds.,
            index 1: (model/h0/attn/c_attn/w:0) [1, 768, 2304]: weight for the self attn for similarity calc in layer 0,
            index2: (model/h0/attn/c_attn/b:0) [2304]: bias for the self attn in layer 0,
            index3: (model/h0/attn/c_proj/w:0) [2304]: weight for the final attn output after softmax of all similarity 
                                                        output in layer 0,
            ... (c_proj/b:0, layer_norm1, mlp, layer_norm2), ... and so on for the remaining 11 layers

        (The clf weight and bias are not assigned since learning hasn't been done yet for it (so just initialized)
        """
        if self.use_encoder is False:
            sess.run([p.assign(ip) for p, ip in zip(tvars[:self.params.n_transfer],
                                                    init_params[:self.params.n_transfer])])
        else:   # load only word embeddings
            # for x in range(len(tvars)):
                # if tvars[x].name == 'model/we:0':
                    # sess.run([p.assign(ip) for p, ip in zip(tvars[x], init_params[0])])
            sess.run([p.assign(ip) for p, ip in zip(tvars[:1], init_params[:1])])

    def load_checkpoint(self, sess, path=None):
        if path is None:
            save_dir = os.path.join(self.params.save_dir, self.params.desc, 'best_params.jl')
        else:
            save_dir = path
        t_vars = utils.find_trainable_variables('model')
        # This should be fine since I'm loading my own saved weights
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

        """
        initially reuse is None (first gpu so not needed), but later reuse = True, so the variable mode/... is shared
        shape - (line 1): [(vocab_size + 3 + 77), 768]. n_ctx: max_seq_length (rocstories = 77, moviecorpus = 511)
              - (line 2): n_embd (768): length of the embedding vector (i.e. e.g. GloVe d=50, but this time, bpe used)
        """
        with tf.variable_scope('model', reuse=reuse):
            # --- generate embedding matrix ---------------------------------------------------------------------------
            we = tf.get_variable(name="we",
                                 shape=[self.params.n_vocab + self.params.n_special + self.params.n_ctx,
                                        self.params.n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            we = dropout(we, self.params.embd_pdrop, train)     # set the dropout for each gpu input

            if self.use_encoder is False:    # Original, just decoder model
                # --- reshape, if not greedy decoding -----------------------------------------------------------------
                # Not fully implemented.
                if not greedy_decoding:
                    # concatenate X12 and x13 (For rocstories: i.e from (?, 2, 77, 2) to (?, 77, 2) ).
                    # Pass '[-1]' to flatten 'X' to several (no of input sequences) 1-D tensors of [77, 2]
                    X = tf.reshape(X, [-1, self.params.n_ctx, self.params.n_embd_d+1])
                    M = tf.reshape(M, [-1, self.params.n_ctx])

                # --- add positional embedding and embed training data ------------------------------------------------
                h_dec = embed(X, we)    # h: embedded input added with pos encoding. All embeddings got from the bpe: we

                # --- decoder stacks ----------------------------------------------------------------------------------
                """
                n_layer = 12, maybe change for my own training
                h_enc: - returns the final output matrix which contains the result after passing through all the 
                            n_layers, mlp, and layer_norm for the input sequence in the current /gpu:X
                       - h shape: [?, 77, 768] :- this is the final layer_norm output to be used for multi-task 
                                            leaarning (e.g. passed to the lm or clf softmax and cross_entropy_loss)
                """
                for layer in range(self.params.n_layer):
                    h_dec = self.dec_block(h_dec, 'h%d' % layer, train=train, scale=True)

            else:   # encoder-decoder model
                # --- reshape, if not greedy decoding -----------------------------------------------------------------
                # Not fully implemented.
                """
                CHECK THE DIFF BETW SPLITTING BEFORE AND AFTER embed() fn below 
                If no diff in split after, then the part before block can be combined with the if cond. above
                
                tf.reshape(): - concatenate X12 and x13 (i.e from (?, 2, 77, 2) to (?, 77, 2) ) for rocstories
                              - Pass '[-1]' to flatten 'X' to several (no of input sequences) 1-D tensors of [77, 2] 
                """
                # X_enc = X[:, 0, 0, :, :]    # # ERROR !!!! Don't only take the first, else problem with q, k
                X_enc = X[:, 0, :, :, :]    # facts and attitudes
                X_dec = X[:, 1, :, :, :]    # dialogues

                if not greedy_decoding:
                    # X_enc needs to be reshaped if only first is NOT taken
                    X_enc = tf.reshape(X_enc, [-1, self.params.n_ctx, self.params.n_embd_d + 1])
                    X_dec = tf.reshape(X_dec, [-1, self.params.n_ctx, self.params.n_embd_d + 1])
                    X = X_dec   # because X is used globally for the lm and clf stage after decoder block
                    M = tf.reshape(M, [-1, self.params.n_ctx])

                # --- add positional embedding and embed training data ------------------------------------------------
                # h: embedded input added with positional encoding. All embeddings got from the bpe: we
                h_enc = embed(X_enc, we)
                h_dec = embed(X_dec, we)
                
                # --- encoder stacks ----------------------------------------------------------------------------------
                for layer in range(self.params.n_enc_layer):
                    h_enc = self.enc_block(h_enc, 'enc_h%d' % layer, train=train, scale=True)

                # Add last layer activation to TBoard
                # tf.summary.histogram('Encoder final attention activations after layer norm', h_enc)

                # --- decoder stacks ----------------------------------------------------------------------------------
                for layer in range(self.params.n_layer):
                    # h = self.dec_block(h_dec, 'h%d' % layer, train=train, scale=True, encoder_output=h_enc)   # ERROR
                    h_dec = self.dec_block(h_dec, 'h%d' % layer, train=train, scale=True, encoder_output=h_enc)

            # --- language modeling loss ------------------------------------------------------------------------------
            """
            tf.reshape(): reshape decoder output to be multiples of n_embd (768) i.e. length of the embedding vector 
                            (i.e. e.g. GloVe d=50, but this time we use bpe instead)
            softmax: for getting prob distribution of the lm_h across dictionary we
            """
            if lm_logits_only:
                lm_h = tf.reshape(h_dec, [-1, self.params.n_embd])
                lm_logits = tf.nn.softmax(tf.matmul(lm_h, we, transpose_b=True))
            else:
                """
                h[:, :-1]: all except the last shape value which then replaced by the roll out (-1). So: [(? * 77), 768]
                tf.matmul(): - perform matmul to get most-similar to the full dictionary tokens as the logits 
                             - shape: [(? * 77), 40558]
                """

                lm_h = tf.reshape(h_dec[:, :-1], [-1, self.params.n_embd])
                lm_logits = tf.matmul(lm_h, we, transpose_b=True)

            """
            labels=: - Why is the first row in the middle from all X removed (even in M below)? Maybe the start token?
                     - for language modeling, the label y would be the input X (reshape the label to be a single vector)
                            but only one of the inputs (The second input (i.e x13)) was used for the modeling since we 
                            aren't performing any classification. 
                     - Note: The label vector contains the start encoding value for both the input token and the 
                            positional encoding for all the input sequences (i.e [1497, :1, :0,  2]) 
                            (So there would be a (1497 * 2) label vector
            tf.reduce_sum()/tf.reduce_sum(): objective fn to maximize sum of the losses but now using the mask to remove
                                            unnecessary positional values from the weight matrix output
            """
            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits,
                                                                       labels=tf.reshape(X[:, 1:, 0], [-1]))
            lm_losses = tf.reshape(lm_losses, [utils.shape_list(X)[0], utils.shape_list(X)[1] - 1])
            lm_losses = (tf.reduce_sum(lm_losses * M[:, 1:], 1) + 1e-9) / (tf.reduce_sum(M[:, 1:], 1) + 1e-9)

            # --- classifier loss -------------------------------------------------------------------------------------
            """
            for rocstories: Y = the label for the right answer (0 or 1 indicating which of the 2 in [1497, 2] from X)
            
            logits: the result from the last layer, 
            loss: the difference between this result and label    
            """
            if self.params.head_type == "clf":
                clf_logits, clf_losses = self.classifier_head(X=X, Y=Y, h=h_dec, train=train)
            elif self.params.head_type == "lm":
                clf_logits, clf_losses = [], []
            else:
                raise ValueError("{} is not a valid parameter for head_type!".format(self.params.head_type))

            return clf_logits, lm_logits, clf_losses, lm_losses

    """
    For rocstories: - X: (?, 2, 77, 2), Y:(?,)  h: (?, 77, 768)  
                    - in X, the first two reps the two combinations of the input sentences and the right and wrong 
                        answers. For the second 2 here, the second column is just the positional encoding for all input 
                        tokens (i.e the input sentences)
    tf.reshape()_1: reshape decoder output (block matrix) to be multiples of n_embd (768). So: clf_h: [(? * 77), 768]
    pool_idx: - tf.equal returns true (which is cast to 1) when index of X is a clf token (along axis 1). The position 
                of the true is then returned as the output. Shape (? * 77). 
              - tf.equal is an element-wise operator so it returns True if any of the elements from X matches the 
                clf_token and false otherwise.
    tf.gather(): - make a [? * 77] matrix (of size 768) + index of clf_token (should also be ?)
                 - not sure what the '+' here does. Does it concatenate the pool_idx matrix with the matrix generated 
                    from range, or does it add the values in matrix pool_idx directly to the values in the range 
                    generated (if it does the latter, doesn't make much sense for the final shape)    
                 - tf.gather gets a the outputs from h using index from X that would be used to perform the 
                    classification and backpropagation based on the indices tf.range... (So we have a list of the values
                    in h containing the first (? * 77) tokens and also the index of the clf or lm tokens
    tf.reshape()_2: reshapes into [?, 2, 768]. clf_pipes: 2 so the first of 2 is the output from clf to be compared with
                    x12 and the second is for x13
    shape: - a 1-D Tensor, representing the shape for randomly generated keep/drop flags.
           - shape[1] = 1: by default, each element is kept or dropped independently, but since the shape is specified 
                            for row = 1, each row will be kept or not kept together
    tf.reshape()_3: after dropout, reshapes into [?, 768], where ? = len_batch_input * 77 (combine x12 and x13)
    clf(): (shape: [?, 1]) returns the result of the flattened fc layer between the clf_h as input and the weight vector
            which are back propagatable (trainable)
    tf.reshape()_4: reshape to [-1, 2]. Since clf_pipes: 2 for (0 or 1) containing the probability of the first or 
                    second input sequence being the right one or not
    softmax_cross_entropy(): - performs a softmax between the outputs of (x12 and x13) and the result is then compared 
                                with the label Y to see if right choice is made or not, and loss is calculated
                             - Y contains a (?) vector of value (0 or 1) telling which of the above (i.e x12 or x13) is 
                                right
    logits: the result from the last layer, 
    loss: the difference between this result and label
    """
    def classifier_head(self, X, Y, h, train=False):
        clf_h = tf.reshape(h, [-1, self.params.n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], self.params.clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(utils.shape_list(X)[0], dtype=tf.int32) * self.params.n_ctx + pool_idx)
        clf_h = tf.reshape(clf_h, [-1, self.params.clf_pipes, self.params.n_embd])
        if train and self.params.clf_pdrop > 0:     # apply dropout if train otherwise not
            shape = utils.shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1 - self.params.clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, self.params.n_embd])
        clf_logits = clf(clf_h, 1, train=train)
        clf_logits = tf.reshape(clf_logits, [-1, self.params.clf_pipes])
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        # tf.summary.histogram('Clf cross_entropy', clf_losses)

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

    """
    scope: hNum where Num: current encoder/decoder layer
    x: embedded input with addition of positional encoding
    nx: 768 for bpe. This gets last element in shape list. (i guess shape_list is done to deal with 'None' in the shape 
        of x, since a tensor op (tf.shape()) is returned instead of None)
    a = attn(): - n_head: 12 multi-head attention
                - use_mask_attn: use the masked multi-head attention only for first decoder attention     
                - returns matmul output of the [softmax(q,k), v] for the attention matrix: [?, 77, 768] which contains 
                    the attn result for each input on itself (q = k)
    norm(): layer normalization with residual addition (for pos embedding rettention). Both input shape: [?, 77, 768]
    encoder_output: differentiate between original decoder model and my modification
    mlp(): Feed-forward. Not sure why 4; maybe for depth?
    """
    def enc_block(self, x, scope, train=False, scale=False):
        with tf.variable_scope(scope):  # scope: enc_h%d
            nx = utils.shape_list(x)[-1]
            a = self.attn(x, 'attn', nx, self.params.n_head, train=train, scale=scale, use_mask_attn=False)
            n = norm(x + a, 'ln_1')
            m = self.mlp(n, 'mlp', nx * 4, train=train)
            h = norm(n + m, 'ln_2')
            return h

    """
    See comment above.
    encoder_output: differentiate between original decoder model and my modification
    """
    def dec_block(self, x, scope, train=False, scale=False, encoder_output=None):
        with tf.variable_scope(scope):  # scope: h%d    (initialized with the loaded params_d.npy values)
            nx = utils.shape_list(x)[-1]
            a = self.attn(x, 'attn', nx, self.params.n_head, train=train, scale=scale)
            n = norm(x + a, 'ln_1')

            if encoder_output is not None:  # encoder-decoder attn performed
                a = self.attn(n, 'attn_enc_dec', nx, self.params.n_head, train=train, scale=scale,
                              encoder_output=encoder_output, use_mask_attn=False)
                n = norm(n + a, 'ln_enc_dec')

            m = self.mlp(n, 'mlp', nx * 4, train=train)
            h = norm(n + m, 'ln_2')
            return h

    """ Multi-layer perceptron
        h = act():  apply linear transformation on the returned (normalized attention matrix) activations
        h2 = conv1d():  perform conv (matmul) using a trainable (back-propagatable) weight on the 4 layer fc output 
    """
    def mlp(self, x, scope, n_state, train=False):
        with tf.variable_scope(scope):
            nx = utils.shape_list(x)[-1]    # nx: 768 for bpe
            act = ACT_FNS[self.params.afn]  # gelu
            h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
            h2 = conv1d(h, 'c_proj', nx, 1, train=train)
            h2 = dropout(h2, self.params.resid_pdrop, train)
            return h2

    def attn(self, x, scope, n_state, n_head, train=False, scale=False, encoder_output=None, use_mask_attn=True):
        assert n_state % n_head == 0    # n_head: attention head; n_state: last element in x shape_list.

        """
        w = tf.matmul(): calculates the similarity measure as an inner product. # [:, 12, x, y] * [:, 12, y, x] would 
                        give us a scalar value at the last two positions indicating the similatity strength. 
                        for rocstories: (x=77, y=64) so final = [?, 12, 77, 77]
        scale: divide by sqrt(dim) to make value not too large
        use_mask_attn: Only perform masked attn for decoder model first multi-head attention
        w = softmax: still same shape but now contains prob. distrib. of the keys that are most similar to the query 
            [?, 12, 77, 77]
        a = tf.matmul(): use the prob. distribution to pick the most similar value, since v: [?, 12, 77, 64], it 
            reshapes back to attn_heads: [?, 12, 77, 64] 
        """
        def _attn(q, k, v, train=False, scale=False, use_mask_attn=use_mask_attn):
            w = tf.matmul(q, k)
            if scale:
                n_state = utils.shape_list(v)[-1]   # last value in v shape_list
                w = w * tf.rsqrt(tf.cast(n_state, tf.float32))
            if use_mask_attn:
                w = mask_attn_weights(w)
            w = tf.nn.softmax(w)
            w = dropout(w, self.params.attn_pdrop, train)
            a = tf.matmul(w, v)
            return a

        """
        scope: hNum where Num: current decoder layer, x: embedded input with addition of positional encoding
        n_state: last element in x shape_list * 3, (768 * 3)
        c_1 = conv1d(): Make trainable weights for q, k, v (i.e. *3) wrt input x (Note: for self-attention, q,k,v are 
                        from the same input)
        tf.split: split into 3 (or 2) along axis 2. If 3, split from 2304 (i.e [77, 768 * 3]) to three tensors each 
                    of [77, 768] representing weights for the q, k, v
        c_2 = conv1d(): Make trainable weights for k, v (i.e. *2) wrt input x (Note: for normal attention, k,v are from 
                        the same input but q is different)
        q = conv1d(): make trainable weights for q
        split_heads(): - each input sequence is now split into 12 attention heads where each attention head deals with 
                        64 input embeddings for an embed_size of 768 (64 * 12 = 768)
                       - diff. last two shapes from q and v which are both similar. k: [?, 12, 64, 77]; 
                                                                                    q, v: [?, 12, 77, 64]
        _attn(): returns the attn matrix but with the most similar value (i.e. after multipyling the softmax 
                    distribution with value, v)
        merge_heads(): returns it to the former shape (i.e. shape it was before it was split into attn: n_heads shape in
                        split_heads()). So now, shape: [?, 77, 768]
        a = conv1d(): returns the output of matmul of the attention and its trainable weights and added bias. 
        """
        with tf.variable_scope(scope):
            if encoder_output is None:  # self-attention model
                c = conv1d(x, 'c_attn', n_state * 3, 1, train=train)
                q, k, v = tf.split(c, 3, 2)
            else:   # using encoder_output as k, v
                c = conv1d(encoder_output, 'c_attn_kv', n_state * 2, 1, train=train)
                k, v = tf.split(c, 2, 2)
                q = conv1d(x, 'c_attn_q', n_state, 1, train=train)

            q = split_heads(q, n_head)
            k = split_heads(k, n_head, k=True)
            v = split_heads(v, n_head)
            a = _attn(q, k, v, train=train, scale=scale, use_mask_attn=use_mask_attn)
            a = merge_heads(a)
            a = conv1d(a, 'c_proj', n_state, 1, train=train)
            a = dropout(a, self.params.resid_pdrop, train)
            return a
