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
    global best_score
    # iter_apply: returns only sum of losses if only_loss=True,
    if LOSS_ONLY:
        tr_cost = iter_apply(data_train['x'][:params.n_valid],      # performs loss calculation only on n_valid: (374) input data
                                        data_train['m'][:params.n_valid],
                                        data_train['y'][:params.n_valid])
        va_cost = iter_apply(data_eval['x'], data_eval['m'], data_eval['y'])    # performs loss calculation only on validation data (which is also of size: n_valid: 374)
    else:   # else returns [concatenated logits, and sum of the losses]
        tr_logits, tr_cost = iter_apply(data_train['x'][:params.n_valid],
                                        data_train['m'][:params.n_valid],
                                        data_train['y'][:params.n_valid],
                                        only_loss=False)
        va_logits, va_cost = iter_apply(data_eval['x'], data_eval['m'], data_eval['y'], only_loss=False)
    # Get average loss for both train data and eval data
    tr_cost = tr_cost/len(data_train['y'][:params.n_valid])
    va_cost = va_cost/params.n_valid
    if params.head_type == "clf":   # logits necessary for clf so we can see the diff btw logits output and actual label
        # accuracy
        tr_acc = skl_metrics.accuracy_score(data_train['y'][:params.n_valid], np.argmax(tr_logits, 1))*100.
        va_acc = skl_metrics.accuracy_score(data_eval['y'], np.argmax(va_logits, 1))*100.
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
            save(os.path.join(params.save_dir, params.desc, 'best_params.jl'))  # save trainable values for best score into /params.desc = 'rocstories'


# apply batch divided data to 4 (or 2) gpus, and the remainder to just one gpu (i guess)
def iter_apply(Xs, Ms, Ys, only_loss=True):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for xmb, mmb, ymb in utils.iter_data(Xs, Ms, Ys, n_batch=params.n_batch_train, truncate=False, verbose=True):   # iter_data divides the input, mask and label data into batches of size: n_batch_train (takes care of fitting input size to batch size)
        n = len(xmb)
        if n == params.n_batch_train:   # divide the batch into no of gpu and get the logits and loss from all 4 (or 2 in my case) gpus
            # logits: the result from the last layer, loss: the difference between this result and label
            res = sess.run([eval_mgpu_logits, eval_mgpu_loss],
                           {X_train: xmb, M_train: mmb, Y_train: ymb})
        else:   # i guess for the last for-loop (last batch were Xs.length % n_batch_train != 0)
            res = sess.run([eval_logits, eval_loss], {X: xmb, M: mmb, Y: ymb})
        if only_loss:
            res = res[1] * n            # [loss*n]
        else:
            res = [r*n for r in res]    # [logits*n, loss*n]
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


def save(path):
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
    parser.add_argument('--submit', action='store_true')    # default = False, but if put in the terminal call, becomes True
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
    params = parser.parse_args()
    print(params)

    # --- set seeds ---------------------------------------------------------------------------------------------------
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)

    # --- logger ------------------------------------------------------------------------------------------------------
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
        dataset = datasets.MovieCorpus(params=params)
        data_train['x'], data_train['m'], data_train['y'], \
            data_eval['x'], data_eval['m'], data_eval['y'] = dataset.prepare_moviecorpus(idx=0,
                                                                                         test=params.eval_best_params)
        data_test['x'] = []
        data_test['m'] = []
    else:
        raise Exception("{}-dataset is not implemented.".format(params.dataset))

    # --- check some parameter dependencies ---------------------------------------------------------------------------
    if params.head_type == "lm" and params.clf_pipes != 1:      # for language modeling, we only want one output, while classification would need 2 since we are multi-tasking (both lm and clf)
        print("WARNING: Only using the language modeling architecture but clf_pipes is {}. clf_pipes is automatically "
              "set to 1! Please check your configuration!".format(params.clf_pipes))
        params.clf_pipes = 1
    if params.dataset == "moviecorpus" and params.n_embd_d != 2:        # n_embd_d is set to 2 because for the movie corpus, we need to embed not just the position but also, the ... (ask Fabian)
        print("WARNING: Using the moviecorpus requires n_embd_d = 2, but is set to {}. It was automatically set to "
              "2! Please check your configuration!".format(params.clf_pipes))
        params.n_embd_d = 2
    if params.n_acc_batch > 1 and params.n_batch_train % params.n_acc_batch != 0:   # ignore for now. Ask Fab again, but default = 1
        raise ValueError("Gradient accumulation active, due to n_acc_batch = {}. n_batch_train is {} which is not "
                         "divisible through n_acc_batch without rest, but must be!")
    elif params.n_acc_batch > 1:
        params.n_batch_train = int(params.n_batch_train / params.n_acc_batch)
        params.gradient_accumulation = True
    else:
        params.gradient_accumulation = False

    # --- generate model as tensorflow graph (train) ------------------------------------------------------------------
    print("Generating model ...")
    transformer_decoder = model.TransformerDecoder(params=params)
    X_train = tf.placeholder(tf.int32, [None, params.clf_pipes, params.n_ctx, params.n_embd_d+1])
    M_train = tf.placeholder(tf.float32, [None, params.clf_pipes, params.n_ctx])
    Y_train = tf.placeholder(tf.int32, [None])

    # returns the result from all four (two in my case) gpus after the training and loss calculated (gradient descent also performed)
    # Just define and add the node, not actually perform train. Actual training is performed in the train loop below
    result = transformer_decoder.mgpu_train(X_train, M_train, Y_train)
    train_step = result[0]
    accumulation_step = result[1]       # None for rocstories   # operation to store the average of the grads from each gpu into a non-trainable tf.Variable of shape tvars
    accumulation_init_step = result[2]  # None for rocstories   # operation to create and assign 0s into a non-trainable tf.Variable of shape tvars
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
    t_vars = utils.find_trainable_variables('model')    # contains the trainable variables from model for gradient descent

    # --- load pretrained parameter -----------------------------------------------------------------------------------
    print("Loading pretrained parameter ...")
    transformer_decoder.init_and_load_parameter_from_file(sess=sess, path="model/")

    # --- add evaluation nodes to tensorflow graph --------------------------------------------------------------------
    # Just add the node, not actually perform eval. Eval is performed in iter_apply,iter_predict
    # perform training but this time turn off dropout???
    eval_mgpu_result = transformer_decoder.mgpu_predict(X_train, M_train, Y_train)  # returns the losses but not grads since only evaluating
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

    # This is necessary for the remaining batch that doesn't fit the batch_size (i.e for the remainder of X.length % n_batch_train != 0)
    X = tf.placeholder(tf.int32, [None, params.clf_pipes, params.n_ctx, params.n_embd_d+1])
    M = tf.placeholder(tf.float32, [None, params.clf_pipes, params.n_ctx])
    Y = tf.placeholder(tf.int32, [None])
    eval_result = transformer_decoder.model(X, M, Y, train=False, reuse=True)   # so no need to divide among the gpus, just use one gpu i guess
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
    with open("params.pkl", 'wb') as f:
        pickle.dump(params, f)

    # --- eval best params and exit -----------------------------------------------------------------------------------
    if params.eval_best_params:
        transformer_decoder.load_checkpoint(sess=sess)
        log(epoch=0, step=0)    # Perform loss and logits calc on train data of size (n_valid) and validation data. Store the accuracy (or perplexity) values and print to see the state of the model based on a new (validation) data
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
            sess.run(accumulation_init_step)    # operation to create and assign 0s into a non-trainable tf.Variable of shape tvars
        for x_batch, m_batch, y_batch in utils.iter_data(*skl_utils.shuffle(data_train['x'],    # iter_data: Basically seperate the data into batches
                                                                            data_train['m'],
                                                                            data_train['y'],
                                                                            random_state=np.random),
                                                         n_batch=params.n_batch_train,
                                                         truncate=True,
                                                         verbose=True):
            if params.gradient_accumulation:
                if step % params.n_acc_batch == params.n_acc_batch - 1:     # after running through the num of grad_acc iteration steps, create a new non-trainable tf.Var and then perform grad calculation and assignment to it
                    sess.run(train_step)
                    sess.run(accumulation_init_step)
                sess.run(accumulation_step, {X_train: x_batch, M_train: m_batch, Y_train: y_batch})     # operation to calc and store the average of the grads from each gpu into a non-trainable tf.Variable of shape tvars
            else:
                # if no grad_acc, directly calculate loss, and also perform train step (grad desc)
                cost, _ = sess.run([loss, train_step], {X_train: x_batch, M_train: m_batch, Y_train: y_batch})  # run loss directly if we aren't accumulating gradients
            step += 1

            # perform evaluation after steps:
            if (step in [1000, 2000, 5000]) and (epoch == 0):   # only perform internal eval on first epoch
                log(epoch=epoch, step=step)
        log(epoch=epoch, step=step)     # perform eval after each epoch

    # After training, if submit, save trainable variables, and then perform prediction
    if params.submit:       # (CHECK THE WEBPAGE FOR THE DEFAULT PARAMS FOR ROCSTORIES)
        sess.run([p.assign(ip) for p, ip in zip(t_vars, joblib.load(os.path.join(params.save_dir,       # store the trainable variables (for the we and the h_layer blocks) as params
                                                                                 params.desc,
                                                                                 'best_params.jl')))])
        predict()
        if params.analysis:
            analysis.rocstories(dataset, params.data_dir,
                                os.path.join(params.submission_dir, FILENAMES[params.dataset]),
                                os.path.join(params.log_dir, 'rocstories.jsonl'))
