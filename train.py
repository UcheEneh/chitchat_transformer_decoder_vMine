import os
import sys
import pickle
import joblib
import random
import argparse
import configparser
import numpy as np
import tensorflow as tf

from sklearn import utils as skl_utils
from sklearn import metrics as skl_metrics

import utils
import model
import datasets
import analysis

# --dataset moviecorpus --desc moviecorpus --n_gpu 1 --use_encoder --submit

# --- definitions -----------------------------------------------------------------------------------------------------
PRED_FNS = {
    'rocstories': lambda x: np.argmax(x, 1),
}
FILENAMES = {
    'rocstories': "ROCStories.tsv",
}
LABEL_DECODERS = {
    'rocstories': None,
}

LOSS_ONLY = False


# --- functions -------------------------------------------------------------------------------------------------------
def log(epoch, step):
    params.n_valid = 300
    global best_score
    """
    iter_apply: returns only sum of losses if only_loss=True
    tr_cost: performs loss calculation only on n_valid: (374) input data
    va_cost: performs loss calculation only on validation data (which is also of size: n_valid: 374)
    
    tr_acc: compare label and argmax of logits outputs for (x12, x13) along axis 1 to tell how many were predicted right
    """
    #
    if LOSS_ONLY:
        tr_cost = iter_apply(data_train['x'][:params.n_valid],
                                        data_train['m'][:params.n_valid],
                                        data_train['y'][:params.n_valid])
        va_cost = iter_apply(data_eval['x'], data_eval['m'], data_eval['y'])
    else:   # else returns [concatenated logits, and sum of the losses]
        tr_logits, tr_cost = iter_apply(data_train['x'][:params.n_valid],
                                        data_train['m'][:params.n_valid],
                                        data_train['y'][:params.n_valid],
                                        only_loss=False)
        # Normally use all validation data, but reduce now for faster eval
        # va_logits, va_cost = iter_apply(data_eval['x'], data_eval['m'], data_eval['y'], only_loss=False)
        va_logits, va_cost = iter_apply(data_eval['x'][:params.n_valid],
                                        data_eval['m'][:params.n_valid],
                                        data_eval['y'][:params.n_valid], only_loss=False)
    # Get average loss for both train data and eval data
    tr_cost = tr_cost/len(data_train['y'][:params.n_valid])
    va_cost = va_cost/params.n_valid
    if params.head_type == "clf":   # logits necessary for clf so we can see the diff btw logits output and actual label
        # accuracy
        tr_acc = skl_metrics.accuracy_score(data_train['y'][:params.n_valid], np.argmax(tr_logits, 1))*100.
        # va_acc = skl_metrics.accuracy_score(data_eval['y'], np.argmax(va_logits, 1)) * 100.
        va_acc = skl_metrics.accuracy_score(data_eval['y'][:params.n_valid], np.argmax(va_logits, 1))*100.
    elif params.head_type == "lm":
        # perplexity
        tr_acc = 2 ** tr_cost
        va_acc = 2 ** va_cost
    else:
        raise ValueError("Not a valid head_type!")
    # Store the accuracy or perplexity values and print
    logger.log(n_epochs=epoch, n_updates=step, tr_cost=tr_cost,
               va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('epoch: %d step: %d tr_cost: %.3f va_cost: %.3f tr_metr: %.2f va_metr: %.2f' % (epoch, step, tr_cost,
                                                                                          va_cost, tr_acc, va_acc))
    if params.submit:
        score = va_acc
        if score > best_score:
            best_score = score
            # save trainable values for best score into /params.desc = 'rocstories'
            save(os.path.join(params.save_dir, params.desc, 'best_params.jl'))


# apply batch divided data to 4 (or 2) gpus, and the remainder to just one gpu (i guess)
def iter_apply(Xs, Ms, Ys, only_loss=True):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    """
    iter_data: divides the input, mask and label data into batches of size: n_batch_train 
                (takes care of fitting input size to batch size)
    for evaluation, n_batch = 16; xmb: for each for-loop: data[h:h+n_batch] for h in [0, 16, 32, ..., 367, (368 - 374)
    
    logits: the result from the last layer, loss: the difference between this result and label
    
    if: divide the batch into no of gpu and get the logits and loss from all 4 (or 2 in my case) gpus
    xmb: shape (for eval): [16, 2, 77, 2]. first 2: x12 and x13 input
    
    else: for the last for-loop []i.e between (368 - 374) (last batch were Xs.length % n_batch_train != 0)
    
    res[1]*n: [loss*n]. multiplied by n to calc the average loss across each batch_size later 
            (since last batch has diff size)
    [r*n for r in res]: [logits*n, loss*n]. logits shape [16, 2] telling us the result from comparing x12 and x13 in 
                        this batch of size 16
    """

    for xmb, mmb, ymb in utils.iter_data(Xs, Ms, Ys, n_batch=params.n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == params.n_batch_train:

            res = sess.run([eval_mgpu_logits, eval_mgpu_loss],
                           {X_train: xmb, M_train: mmb, Y_train: ymb})
        else:
            res = sess.run([eval_logits, eval_loss], {X: xmb, M: mmb, Y: ymb})
        if only_loss:
            res = res[1] * n
        else:
            res = [r*n for r in res]
        results.append(res)
    if only_loss:
        return np.sum(results)      # return only the sum of losses
    else:
        results = zip(*results)
        return [fn(res) for res, fn in zip(results, fns)]   # concatenate logits and sum the losses


def iter_predict(Xs, Ms):
    logits = []
    for xmb, mmb in utils.iter_data(Xs, Ms, n_batch=params.n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == params.n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train: xmb, M_train: mmb}))
        else:
            logits.append(sess.run(eval_logits, {X: xmb, M: mmb}))
    logits = np.concatenate(logits, 0)      # logits: the result from the last layer
    return logits


# New method from Fabian
'''
EVAL_FNS = {
    'lm_ppl': lambda x: compute_lm_ce(x),
    'clf_acc': lambda x: compute_clf_acc(x)
}


def compute_lm_ce(res):
    """ Returns the sum of the cross-entropy values and the number of values.

    The function ignores values = 1.0
    This is due to wrong utterances (which have 0-masks).
    """
    sum = 0
    n_values = 0
    for value in res.tolist():
        if value != 1.0:
            sum += value
            n_values += 1
    return sum, n_values


def compute_clf_acc(res):
    """ Computes the sum of the values and the number of values. """
    sum = 0
    res_as_list = res.tolist()
    for value in res_as_list:
        sum += value
    return sum, len(res_as_list)


# --- functions -------------------------------------------------------------------------------------------------------
def log(epoch, step):
    """ Computes metrics and updates best params.
    """
    global best_score

    # --- compute metrics on train and eval set ---
    metrics_train = compute_metrics(data_train['x'][:params.n_valid],
                                    data_train['m'][:params.n_valid],
                                    data_train['y'][:params.n_valid])
    metrics_eval = compute_metrics(data_eval['x'], data_eval['m'], data_eval['y'])

    # --- compute mean values (and prepare for print) ---
    values_to_print = []
    for metrics, label in zip([metrics_train, metrics_eval], ["tr", "va"]):
        for metric, values in metrics.items():
            value = values['val'] / values['num']
            values_to_print.append({"name": label + "_" + metric, "value": value})

    # --- print ---
    string = "epoch: {} step: {}".format(epoch, step)
    for value in values_to_print:
        string += " {}: {}".format(value["name"], round(value["value"], 2))
    print(string)

    # --- update best params if needed ---
    if params.submit:
        if params.best_epoch_criteria == "clf":
            score = metrics_eval['clf_acc']['val'] / metrics_eval['clf_acc']['num']
        elif params.best_epoch_criteria == "lm":
            score = metrics_eval['lm_ppl']['val'] / metrics_eval['lm_ppl']['num']
        else:
            raise Exception("")
        if score > best_score:
            best_score = score
            save(os.path.join(params.save_dir, params.desc, 'best_params.jl'))


def compute_metrics(Xs, Ms, Ys):
    """ Dynamically computes metrics.

    Available metrics:
        language modeling loss (cross-entropy)
        classification loss (accuracy)

    Returns:
        A dict.     The results. Description to be done.
    """
    # --- compute available metrics ---
    available_metrics = ["lm_ppl"]
    if params.head_type == "clf":
        available_metrics.append("clf_acc")

    # --- lookup corresponding tensors ---
    tensors_mgpu = []
    tensors_single = []
    for metric in available_metrics:
        if "lm" in metric:
            tensors_mgpu.append(eval_mgpu_lm_losses)
            tensors_single.append(eval_lm_losses)
        elif "clf" in metric:
            tensors_mgpu.append(eval_mgpu_clf_losses)
            tensors_single.append(eval_clf_losses)

    # --- compute metrics ---
    results_summed = {}
    for xmb, mmb, ymb in utils.iter_data(Xs, Ms, Ys, n_batch=params.n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == params.n_batch_train:
            results = sess.run(tensors_mgpu, {X_train: xmb, M_train: mmb, Y_train: ymb})
        else:
            results = sess.run(tensors_single, {X: xmb, M: mmb, Y: ymb})

        for res, metric in zip(results, available_metrics):
            val, num = EVAL_FNS[metric](res)
            if metric in results_summed:
                results_summed[metric]["val"] += val
                results_summed[metric]["num"] += num
            else:
                results_summed[metric] = {
                    "val": val,
                    "num": num
                }

    return results_summed
'''


def predict():
    filename = FILENAMES[params.dataset]    # ROCStories.tsv
    pred_fn = PRED_FNS[params.dataset]      # fn to return the argmax between (logits, 1)
    label_decoder = LABEL_DECODERS[params.dataset]
    predictions = pred_fn(iter_predict(data_test['x'], data_test['m']))     # apply test data to model
    if label_decoder is not None:   # None for rocstories
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(params.submission_dir, filename)    # submission/ROCStories.tsv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


def save(path):     # save the values of the trainable variables (we, h0, h1, ...)
    ps = sess.run(utils.find_trainable_variables('model'))
    joblib.dump(ps, utils.make_path(path))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
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
    parser.add_argument('--submit', action='store_true')    # default = False, but if put in the terminal call, = True
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=defaults['seed'])
    parser.add_argument('--n_iter', type=int, default=defaults['n_iter'])
    parser.add_argument('--n_batch', type=int, default=defaults['n_batch'])
    parser.add_argument('--n_acc_batch', type=int, default=defaults['n_acc_batch'])
    parser.add_argument('--max_grad_norm', type=int, default=defaults['max_grad_norm'])
    parser.add_argument('--only_last_utterance_masking', type=bool, default=defaults['only_last_utterance_masking'])
    parser.add_argument('--dynamic_pos_embeddings', type=bool, default=defaults['dynamic_pos_embeddings'])
    parser.add_argument('--begin_cut_dialogues', type=str2bool, default=defaults['begin_cut_dialogues'])
    parser.add_argument('--lr', type=float, default=defaults['lr'])
    parser.add_argument('--lr_warmup', type=float, default=defaults['lr_warmup'])
    parser.add_argument('--n_ctx', type=int, default=defaults['n_ctx'])
    parser.add_argument('--n_embd', type=int, default=defaults['n_embd'])
    parser.add_argument('--n_embd_d', type=int, default=defaults['n_embd_d'])
    parser.add_argument('--n_head', type=int, default=defaults['n_head'])
    parser.add_argument('--n_layer', type=int, default=defaults['n_layer'])
    parser.add_argument('--head_type', type=str, default=defaults['head_type'])
    parser.add_argument('--embd_pdrop', type=float, default=defaults['embd_pdrop'])
    parser.add_argument('--attn_pdrop', type=float, default=defaults['attn_pdrop'])
    parser.add_argument('--resid_pdrop', type=float, default=defaults['resid_pdrop'])
    parser.add_argument('--clf_pdrop', type=float, default=defaults['clf_pdrop'])
    parser.add_argument('--clf_pipes', type=int, default=defaults['clf_pipes'])
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
    parser.add_argument('--eval_best_params', action='store_true')
    parser.add_argument('--dict_error_fix', type=bool, default=defaults['dict_error_fix'])
    parser.add_argument('--use_encoder', action='store_true')   # default = False, but if put in the terminal, = True
    params = parser.parse_args()
    print(params)

    # --- set seeds ---------------------------------------------------------------------------------------------------
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    # --- logger ------------------------------------------------------------------------------------------------------
    # save default params
    logger = utils.ResultLogger(path=os.path.join(params.log_dir, '{}.jsonl'.format(params.desc)), **params.__dict__)

    # --- load and prepare train data ---------------------------------------------------------------------------------
    data_train = {}
    data_eval = {}
    data_test = {}
    dataset = None
    if params.dataset == "rocstories":
        dataset = datasets.Rocstories(params=params)
        data_train['x'], data_train['m'], data_train['y'], \
            data_eval['x'], data_eval['m'], data_eval['y'], \
            data_test['x'], data_test['m'] = dataset.prepare_rocstories()
    elif params.dataset == "moviecorpus":
        # more like load "or" prepare train data, because this fn is called again during train
        dataset = datasets.MovieCorpus(params=params)
        data_train['x'], data_train['m'], data_train['y'], \
            data_eval['x'], data_eval['m'], data_eval['y'] = dataset.prepare_moviecorpus(idx=0,
                                                                                         test=params.eval_best_params)
        data_test['x'] = []
        data_test['m'] = []
    else:
        raise Exception("{}-dataset is not implemented.".format(params.dataset))

    # --- check some parameter dependencies ---------------------------------------------------------------------------
    # for lm, we only want one output, while classification would need 2 since we are multi-tasking (both lm and clf)
    if params.head_type == "lm" and params.clf_pipes != 1:
        print("WARNING: Only using the language modeling architecture but clf_pipes is {}. clf_pipes is automatically "
              "set to 1! Please check your configuration!".format(params.clf_pipes))
        params.clf_pipes = 1
    if params.dataset == "moviecorpus" and params.n_embd_d != 2:
        print("WARNING: Using the moviecorpus requires n_embd_d = 2, but is set to {}. It was automatically set to "
              "2! Please check your configuration!".format(params.clf_pipes))
        params.n_embd_d = 2
    if params.n_acc_batch > 1 and params.n_batch_train % params.n_acc_batch != 0:
        raise ValueError("Gradient accumulation active, due to n_acc_batch = {}. n_batch_train is {} which is not "
                         "divisible through n_acc_batch without rest, but must be!")
    elif params.n_acc_batch > 1:
        params.n_batch_train = int(params.n_batch_train / params.n_acc_batch)
        params.gradient_accumulation = True
    else:
        params.gradient_accumulation = False
    if params.use_encoder:
        params.n_enc_layer = 6  # to make this editable

    # --- generate model as tensorflow graph (train) ------------------------------------------------------------------
    print("Generating model ...")
    transformer_decoder = model.Transformer(params=params)
    if params.use_encoder is False:     # original decoder model
        X_train = tf.placeholder(tf.int32, [None, params.clf_pipes, params.n_ctx, params.n_embd_d+1])
    else:   # with encoder-decoder model
        X_train = tf.placeholder(tf.int32, [None, 2, params.clf_pipes, params.n_ctx, params.n_embd_d + 1])
    M_train = tf.placeholder(tf.float32, [None, params.clf_pipes, params.n_ctx])
    Y_train = tf.placeholder(tf.int32, [None])

    """
    This just defines and adds the node, not perform actual training. Training is performed in the train loop below
    - returns the result from all four (two) gpus after training and loss calculated (gradient descent also performed)
    
    accumulation_step: op to store the avg of the grads from each gpu into a non-trainable tf.Variable of shape tvars
    accumulation_init_step: op to create and assign 0s into a non-trainable tf.Variable of shape tvars
    """
    result = transformer_decoder.mgpu_train(X_train, M_train, Y_train, use_encoder=params.use_encoder)
    train_step = result[0]
    accumulation_step = result[1]       # None for rocstories
    accumulation_init_step = result[2]  # None for rocstories
    if params.head_type == "clf":
        logits = result[3]      # shape: [?, 2] *Note: 2 for classifying the input as the right or wrong
        clf_loss = result[4]    # shape: [?]    * label - predicted_logit
        lm_loss = result[5]     # shape: [?]
        loss = clf_loss
    elif params.head_type == "lm":
        lm_loss = result[3]
        loss = lm_loss
    else:
        raise ValueError("Not a valid head_type!")
    config = tf.ConfigProto()       # Tensorflow properties (ask Fabian)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    t_vars = utils.find_trainable_variables('model')    # contains the trainable variables from model for gradient desc

    # --- load pretrained parameter -----------------------------------------------------------------------------------
    print("Loading pretrained parameter ...")
    transformer_decoder.init_and_load_parameter_from_file(sess=sess, path="model/", use_encoder=params.use_encoder)

    # --- add evaluation nodes to tensorflow graph --------------------------------------------------------------------
    # Just add the node, not actually perform eval. Eval is performed in iter_apply,iter_predict
    # perform training but this time turn off dropout???
    # eval_mgpu_result: returns the losses but not grads since only evaluating
    eval_mgpu_result = transformer_decoder.mgpu_predict(X_train, M_train, Y_train, use_encoder=params.use_encoder)

    """
    if params.head_type == "clf":
        eval_mgpu_clf_logits = eval_mgpu_result[0]
        eval_mgpu_clf_losses = eval_mgpu_result[1]
        eval_mgpu_clf_loss = tf.reduce_mean(eval_mgpu_clf_losses)
        eval_mpgu_lm_logits = None
        eval_mgpu_lm_losses = eval_mgpu_result[2]
        eval_mgpu_lm_loss = tf.reduce_mean(eval_mgpu_lm_losses)
    elif params.head_type == "lm":
        eval_mgpu_clf_logits = None
        eval_mgpu_clf_losses = None
        eval_mgpu_clf_loss = None
        eval_mgpu_lm_logits = eval_mgpu_result[0]
        eval_mgpu_lm_losses = eval_mgpu_result[1]
        eval_mgpu_lm_loss = tf.reduce_mean(eval_mgpu_lm_losses)
    """

    if params.head_type == "clf":
        eval_mgpu_logits = eval_mgpu_result[0]
        eval_mgpu_clf_losses = eval_mgpu_result[1]
        eval_mgpu_lm_losses = eval_mgpu_result[2]
        eval_mgpu_loss = tf.reduce_mean(eval_mgpu_clf_losses)
    elif params.head_type == "lm":
        eval_mgpu_logits = eval_mgpu_result[0]
        eval_mgpu_losses = eval_mgpu_result[1]
        eval_mgpu_loss = tf.reduce_mean(eval_mgpu_losses)
    else:
        raise ValueError("Not a valid head_type!")

    """
    This is necessary for the remaining batch that doesn't fit the batch_size (i.e for the remainder of X.length % n_batch_train != 0)
    So call model() directly, since no need to divide among the gpus, just use one gpu i guess
    """
    if params.use_encoder is False:     # original decoder model
        X = tf.placeholder(tf.int32, [None, params.clf_pipes, params.n_ctx, params.n_embd_d+1])
    else:   # encoder-decoder model
        X = tf.placeholder(tf.int32, [None, 2, params.clf_pipes, params.n_ctx, params.n_embd_d + 1])
    M = tf.placeholder(tf.float32, [None, params.clf_pipes, params.n_ctx])
    Y = tf.placeholder(tf.int32, [None])
    eval_result = transformer_decoder.model(X, M, Y, train=False, reuse=True, use_encoder=params.use_encoder)
    eval_clf_logits = eval_result[0]
    eval_lm_logits = eval_result[1]
    eval_clf_losses = eval_result[2]
    eval_lm_losses = eval_result[3]

    if params.head_type == "clf":
        eval_logits = eval_clf_logits
        eval_loss = tf.reduce_mean(eval_clf_losses)
    elif params.head_type == "lm":
        eval_logits = eval_lm_logits
        eval_loss = tf.reduce_mean(eval_lm_losses)
    else:
        raise ValueError("Not a valid head_type!")
    """
    eval_lm_loss = tf.reduce_mean(eval_lm_losses)
    if params.head_type == "clf":
        eval_clf_loss = tf.reduce_mean(eval_clf_losses)
    else:
        eval_clf_loss = None
    """

    with open("params.pkl", 'wb') as f:
        pickle.dump(params, f)

    # --- eval best params and exit -----------------------------------------------------------------------------------
    if params.eval_best_params:
        transformer_decoder.load_checkpoint(sess=sess)
        # Perform loss and logits calc on train data of size (n_valid) and validation data.
        # Store the accuracy (or perplexity) vals and print to see the state of the model based on a new (validtn) data
        log(epoch=0, step=0)
        sys.exit()

    # --- train loop --------------------------------------------------------------------------------------------------
    print("Start training ...")
    step = 0
    best_score = 0
    log(epoch=0, step=0)    # perform evaluation on first step
    for epoch in range(params.n_iter):  # n_iter = 3
        # Probably not needed for rocstories
        if epoch > 0 and params.head_type == "clf" and params.dataset == "moviecorpus":
            data_train['x'], data_train['m'], data_train['y'], \
            data_eval['x'], data_eval['m'], data_eval['y'] = dataset.prepare_moviecorpus(idx=epoch % 2)
            if params.gradient_accumulation:
                params.n_batch_train = int(params.n_batch_train / params.n_acc_batch)

        if params.gradient_accumulation:    # False for rocstories
            sess.run(accumulation_init_step)    # op to assign 0s to a non-trainable tf.Variable of shape tvars

        """
        skl_utils.shuffle: Shuffle arrays or sparse matrices in a consistent way, so data_trains: ['x'], ['m'], ['y']
                            would always be arranged in the right order  
                            
        A - For rocstories with n_batch_train = 16, gradient_acc = False, so n_acc_batch = 1;
            at each for-loop, since 1497 /16 = 93.3, perform 93 operations on the data each of size 16, 
            (Note: since iter_data is actually a list of these 93 separated different input data)
            
        - iter_data performs a yield(d[i : i+n_batch)) for X, M, and Y, Now n_batch = n_batch_train = 16
        - iter_data: Basically separate the data into batches list (so for train: have a total of (1497 / 16) = 93.xx)
        - so train batch i: [16, 32, ..., 1488, (1489 - 1497)] (total length = 94)
        - meaning the for-loop would be:
            - at current_batch 0, data[0:15]
            - at current_batch 1, data[16:31]
            - ...
            - at current_batch 92, data[1472:1488]
            - at current_batch 93, data[1489:1497]
            
            
        B - For moviecorpus with gradient_acc = true, n_gpu = 2, n_batch=8 (so total batch size actually 2*8=16), 
            n_acc_batch=4, n_batch_train = 16/4 = 4. 
            
        - iter_data performs a yield(d[i : i+n_batch)) for X, M, and Y. Now n_batch = n_batch_train = 4
            But since it is shared among two gpus during training, actually = 2
        - iter_data: Basically separate the data into batches list (so for train: have a total of (69799 / 4) = 17499.xx)
            But since using across 2 gpus: total after one loop: 34,899 operations
        - so train batch i: [4, 8, ..., 69796, (69797 - 69799)] (total length = 17500)
        - meaning the for-loop would be:
            - at current_batch 0, x_batch: data[0:3]
            - at current_batch 1, x_batch: data[3:7]
            - ...
            - at current_batch 17448, x_batch: data[69791:69795]
            - at current_batch 17449, x_batch: data[69795:69799]    # last 3 batches  
        """
        for current_batch, x_batch, m_batch, y_batch in enumerate(utils.iter_data(*skl_utils.shuffle(data_train['x'],
                                                                                             data_train['m'],
                                                                                             data_train['y'],
                                                                                             random_state=np.random),
                                                                                  n_batch=params.n_batch_train,
                                                                                  truncate=True,
                                                                                  verbose=True)):
            """
            if grad_acc:
                - if equal: after running through the num of grad_acc iteration steps, create a new non-trainable tf.Var 
                        and then perform grad calculation and assignment to it
                - run(train_step): perform grad update
                - run(acc_init_step): clear the tvars after gradient update
                - run(acc_step): operation to calc and store the average of the grads from each gpu into a non-trainable
                                tf.Variable of shape tvars

            else:
                - run: calc loss directly if we aren't accumulating gradients and also perform train step (grad desc)
                - x_batch: shape (for train): [16, 2, 77, 2]     # first 2: x12 and x13 input

            step + 1: step is increased after each loss calculated regardless of batch or epoch being run 
                        (For rocstories: so after every 16 input sequence, step += 1)
            
            log()_1: only perform internal eval on first epoch (important epoch since gradient update changing faster) 
                    (For moviecorpus. rocstories corpus not that big)
            log()_2: perform eval after each epoch
            """
            if params.gradient_accumulation:
                if step % params.n_acc_batch == params.n_acc_batch - 1:
                    sess.run(train_step)
                    sess.run(accumulation_init_step)
                sess.run(accumulation_step, {X_train: x_batch, M_train: m_batch, Y_train: y_batch})
            else:
                cost, _ = sess.run([loss, train_step], {X_train: x_batch, M_train: m_batch, Y_train: y_batch})
            step += 1

            # perform evaluation after steps:
            if (step in [100, 1000, 2000, 5000]) and (epoch == 0):
                log(epoch=epoch, step=step)
            print("-------- May be wrong format ------- Current mini-batch {0} in Batch {1} performed each across "
                  "{2} gpus in Epoch {3} done".format(current_batch, params.n_batch_train, params.n_gpu, epoch))
        log(epoch=epoch, step=step)
        print("**************** Epoch {0}/{1} done".format(epoch, params.n_iter))

    # After training, if submit, save trainable variables, and then perform prediction
    if params.submit:
        # store the trainable variables (for the we and the h_layer blocks) as params
        sess.run([p.assign(ip) for p, ip in zip(t_vars, joblib.load(os.path.join(params.save_dir,
                                                                                 params.desc,
                                                                                 'best_params.jl')))])
        if params.dataset == "rocstories":
            predict()
        if params.analysis:
            analysis.rocstories(dataset, params.data_dir,
                                os.path.join(params.submission_dir, FILENAMES[params.dataset]),
                                os.path.join(params.log_dir, 'rocstories.jsonl'))
