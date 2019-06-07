import os
import csv
import glob
import json
import copy
import pickle
import numpy as np

import utils
import text_utils
from tqdm import tqdm

from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

SEED = 3535999445

OVERWRITE_EXISTING_DATA = False  # If True, the system does not load post processed files.


class MovieCorpus:
    def __init__(self, params):
        self.params = params
        print("PARAMS:")
        print(self.params)
        # self.delex_dict = {
        #     'movie#0': '[w_movie]',
        #     'actor#0': '[w_actor0]',
        #     'actor#1': '[w_actor1]',
        #     'actor#2': '[w_actor2]',
        #     'director#0': '[w_director]',
        #     'writer#0': '[w_writer]',
        #     'year#0': '[w_year]',
        #     'budget#0': '[w_budget]',
        #     'certificate#0': '[w_certificate]',
        #     'country#0': '[w_country]',
        #     'genre#0': '[w_genre0]',
        #     'genre#1': '[w_genre1]',
        #     'end#t': '[end]',
        # }
        self.delex_dict = OrderedDict([
            ('movie#0', '[w_movie]'),
            ('actor#0', '[w_actor0]'),
            ('actor#1', '[w_actor1]'),
            ('actor#2', '[w_actor2]'),
            ('director#0', '[w_director]'),
            ('writer#0', '[w_writer]'),
            ('year#0', '[w_year]'),
            ('budget#0', '[w_budget]'),
            ('certificate#0', '[w_certificate]'),
            ('country#0', '[w_country]'),
            ('genre#0', '[w_genre0]'),
            ('genre#1', '[w_genre1]'),
            ('end#t', '[end]'),
        ])

        # self.fact_dict = {
        #     'movie#0': '[pos_f_movie]',
        #     'actor#0': '[pos_f_actor0]',
        #     'actor#1': '[pos_f_actor1]',
        #     'actor#2': '[pos_f_actor2]',
        #     'director#0': '[pos_f_director]',
        #     'writer#0': '[pos_f_writer]',
        #     'plot': '[pos_f_plot]',
        # }
        self.fact_dict = OrderedDict([
            ('movie#0', '[pos_f_movie]'),
            ('actor#0', '[pos_f_actor0]'),
            ('actor#1', '[pos_f_actor1]'),
            ('actor#2', '[pos_f_actor2]'),
            ('director#0', '[pos_f_director]'),
            ('writer#0', '[pos_f_writer]'),
            ('plot', '[pos_f_plot]'),
        ])

        # for downward compatibility
        if not hasattr(self.params, 'dict_error_fix'):
            self.params.dict_error_fix = False

        if not self.params.dict_error_fix:
            self.attitude_dict = {
                'movie#0': '[pos_a_movie]',
                'actor#0': '[pos_a_actor0]',
                'actor#1': '[pos_a_actor1]',
                'actor#2': '[pos_a_actor2]',
                'director#0': '[pos_a_director]',
                'writer#0': '[pos_a_writer]',
                'country#0': '[pos_a_plot]',
                'genre#0': '[pos_a_plot]',
                'genre#1': '[pos_a_plot]',
                'budget#0': '[pos_a_budget]',
                'certificate#0': '[pos_a_certificate]',
            }
        else:
            # self.attitude_dict = {
            #     'movie#0': '[pos_a_movie]',
            #     'actor#0': '[pos_a_actor0]',
            #     'actor#1': '[pos_a_actor1]',
            #     'actor#2': '[pos_a_actor2]',
            #     'director#0': '[pos_a_director]',
            #     'writer#0': '[pos_a_writer]',
            #     'country#0': '[pos_a_country]',
            #     'genre#0': '[pos_a_genre0]',
            #     'genre#1': '[pos_a_genre1]',
            #     'budget#0': '[pos_a_budget]',
            #     'certificate#0': '[pos_a_certificate]',
            # }
            self.attitude_dict = OrderedDict([
                ('movie#0', '[pos_a_movie]'),
                ('actor#0', '[pos_a_actor0]'),
                ('actor#1', '[pos_a_actor1]'),
                ('actor#2', '[pos_a_actor2]'),
                ('director#0', '[pos_a_director]'),
                ('writer#0', '[pos_a_writer]'),
                ('country#0', '[pos_a_country]'),
                ('genre#0', '[pos_a_genre0]'),
                ('genre#1', '[pos_a_genre1]'),
                ('budget#0', '[pos_a_budget]'),
                ('certificate#0', '[pos_a_certificate]'),
            ])

        self.text_encoder = text_utils.TextEncoder(params.encoder_path, params.bpe_path, delex_dict=self.delex_dict)
        self.encoder = self.text_encoder.encoder
        num_of_words = len(self.encoder)
        print("Number of Words:")
        print(num_of_words)

        self.params.n_vocab = len(self.text_encoder.encoder)
        print("N Vocab:")
        print(self.params.n_vocab)

        for key in self.delex_dict:
            self.encoder[self.delex_dict[key]] = len(self.encoder)

        self.encoder['[start]'] = len(self.encoder)  # TODO: really needed?
        # self.encoder['[end]'] = len(self.encoder)  # already added via delex_dict
        self.encoder['[classifier]'] = len(self.encoder)
        self.params.clf_token = self.encoder['[classifier]']

        for i in range(6):
            self.encoder['[att_{}]'.format(i)] = len(self.encoder)

        for key in self.fact_dict:
            self.encoder[self.fact_dict[key]] = len(self.encoder)

        for key in self.attitude_dict:
            self.encoder[self.attitude_dict[key]] = len(self.encoder)

        self.encoder['[pos_human]'] = len(self.encoder)
        self.encoder['[pos_bot]'] = len(self.encoder)

        # kind of hacky, but works for the moment ...
        self.text_encoder.encoder = self.encoder
        self.text_encoder.update_decoder()

        self.params.n_special = len(self.encoder) - num_of_words

        self.dialogues_src = []

    def prepare_moviecorpus(self, idx=0, test=False):
        # --- load data -----------------------------------------------------------------------------------------------
        if test:
            logfiles = glob.glob(os.path.join(self.params.data_dir, "moviecorpus/raw_test/*"))
            processed_file_names = ["moviecorpus_test.pkl"]
        else:
            logfiles = glob.glob(os.path.join(self.params.data_dir, "moviecorpus/raw/*"))
            if self.params.head_type == "lm":
                processed_file_names = ["moviecorpus_lm_prepared.pkl"]
            elif self.params.head_type == "clf":
                processed_file_names = ["moviecorpus_clf_prepared_0.pkl",
                                        "moviecorpus_clf_prepared_1.pkl",
                                        "moviecorpus_clf_prepared_eval.pkl"]
            else:
                raise Exception("No valid head-type.")
        processed_files_path = []
        for processed_file_name in processed_file_names:
            processed_files_path.append(os.path.join(self.params.data_dir, "moviecorpus", processed_file_name))

        # --- check for some (newer) parameter ------------------------------------------------------------------------
        # This could be removed later, if no old models are used anymore.
        if not hasattr(self.params, 'dynamic_pos_embeddings'):
            self.params.dynamic_pos_embeddings = False
        if not hasattr(self.params, 'begin_cut_dialogues'):
            self.params.begin_cut_dialogues = False
        if not hasattr(self.params, 'only_last_utterance_masking'):
            self.params.only_last_utterance_masking = False

        # --- restore data (if True) or postprocess the raw data (if False) -------------------------------------------
        file_missing = False
        for processed_file_path in processed_files_path:
            if not os.path.exists(processed_file_path):
                file_missing = True

        if not file_missing and not OVERWRITE_EXISTING_DATA:
            print("INFO: Loading prepared data.")
            if self.params.head_type == "clf":
                # --- classifier data load ----------------------------------------------------------------------------
                # For a fixed size of epochs, different "wrong utterances" are available. To load a specific
                # set of data, use the "idx" parameter from this function.
                # The last file is always the evaluation set, which is the same for every epoch.
                with open(processed_files_path[idx], 'rb') as f:
                    data = pickle.load(f)
                x_train = data['x_train']
                m_train = data['m_train']
                y_train = data['y_train']
                with open(processed_files_path[-1], 'rb') as f:
                    data = pickle.load(f)
                x_eval = data['x_eval']
                m_eval = data['m_eval']
                y_eval = data['y_eval']
            else:
                # --- language model only data load -------------------------------------------------------------------
                if idx != 0:
                    print("WARNING: Parameter 'idx' of function 'prepare_moviecorpus' has no effect in this "
                          "configuration!")
                with open(processed_files_path[0], 'rb') as f:
                    data = pickle.load(f)
                x_train = data['x_train']
                m_train = data['m_train']
                y_train = data['y_train']
                x_eval = data['x_eval']
                m_eval = data['m_eval']
                y_eval = data['y_eval']
        else:
            for logfile in logfiles:
                with open(logfile, 'r') as f:
                    self.dialogues_src.append(json.load(f))
            x_train, m_train, y_train, x_eval, m_eval, y_eval = self.postprocess(self.dialogues_src)

            if not test:
                if self.params.head_type == "clf":
                    with open(processed_files_path[0], 'wb') as f:
                        pickle.dump({
                            'x_train': x_train[0],
                            'm_train': m_train[0],
                            'y_train': y_train[0],
                        }, f)
                    with open(processed_files_path[1], 'wb') as f:
                        pickle.dump({
                            'x_train': x_train[1],
                            'm_train': m_train[1],
                            'y_train': y_train[1],
                        }, f)
                    with open(processed_files_path[2], 'wb') as f:
                        pickle.dump({
                            'x_eval': x_eval,
                            'm_eval': m_eval,
                            'y_eval': y_eval,
                        }, f)
                else:
                    with open(processed_files_path[0], 'wb') as f:
                        pickle.dump({
                            'x_train': x_train,
                            'm_train': m_train,
                            'y_train': y_train,
                            'x_eval': x_eval,
                            'm_eval': m_eval,
                            'y_eval': y_eval,
                        }, f)
            x_train = x_train[idx]
            m_train = m_train[idx]
            y_train = y_train[idx]

        # --- compute some statistics for training --------------------------------------------------------------------
        # if self.params.head_type == "lm":
        #     x_train = [x_train]
        #     m_train = [m_train]
        #     y_train = [y_train]
        self.params.n_ctx = x_train[0].shape[1]
        self.params.n_train = len(x_train)
        self.params.n_valid = len(x_eval)
        self.params.n_batch_train = self.params.n_batch * self.params.n_gpu
        self.params.n_updates_total = (self.params.n_train // self.params.n_batch_train) * self.params.n_iter

        return x_train, m_train, y_train, x_eval, m_eval, y_eval

    def postprocess(self, dialogues_src):
        # --- tokenizing and byte-pair encoding -----------------------------------------------------------------------
        x_dialogues = []
        t_dialogues = []
        m_dialogues = []
        x_wr_dialogues = []
        x_facts = []
        t_facts = []
        b_facts = []
        x_attitudes = []
        t_attitudes = []

        # FOR DEBUG
        log_count = 0
        for dialogue_src in dialogues_src:
            # try:
            dialogue = dialogue_src['dialogue_ner']
            dialogue = self.process_dialogue(dialogue=dialogue)
            facts = dialogue_src['facts']
            attitudes = dialogue_src['attitudes']
            speaker_turns = len(dialogue) - 1
            if self.params.head_type == "clf":
                self.process_dialogues(dialogue=dialogue, x=x_dialogues, t=t_dialogues, m=m_dialogues,
                                       x_wr=x_wr_dialogues, dialogue_src=dialogue_src)
            else:
                self.process_dialogues(dialogue=dialogue, x=x_dialogues, t=t_dialogues, m=m_dialogues)
            self.process_facts(facts=facts, speaker_turns=speaker_turns, x=x_facts, t=t_facts, b=b_facts)
            self.process_attitudes(attitudes=attitudes, speaker_turns=speaker_turns, x=x_attitudes, t=t_attitudes)
            # if len(x_attitudes) > 500:
            #     stop = "here"
            #     break
            # except Exception as e:
            #     stop = "here"

        # --- compute maxlen ------------------------------------------------------------------------------------------
        max_length = 0
        idx_to_skip = []  # mainly because they are too long. Other solution: Cut the beginning of the dialogue.
        for i, (d, f, a) in enumerate(zip(x_dialogues, x_facts, x_attitudes)):
            skip = False
            if self.params.head_type == "clf":
                for d_wrong in x_wr_dialogues[i]:
                    this_length = len(d_wrong) + len(f) + len(a)
                    if this_length > self.params.n_ctx - 1:
                        idx_to_skip.append(i)
                        skip = True
                        continue
                    elif this_length > max_length:
                        max_length = this_length
            if skip:
                continue
            this_length = len(d) + len(f) + len(a)
            if this_length > self.params.n_ctx - 1:
                idx_to_skip.append(i)
            elif this_length > max_length:
                max_length = this_length
        self.params.n_ctx = max_length + 1
        print("Processed {} dialogues into {} samples. {} were too long.".format(len(dialogues_src),
                                                                                 len(x_dialogues),
                                                                                 len(idx_to_skip)))
        print("max length = {}".format(self.params.n_ctx))
        x_dialogues = MovieCorpus._multi_pop(x_dialogues, idx_to_skip)
        x_facts = MovieCorpus._multi_pop(x_facts, idx_to_skip)
        x_attitudes = MovieCorpus._multi_pop(x_attitudes, idx_to_skip)
        x_wr_dialogues = MovieCorpus._multi_pop(x_wr_dialogues, idx_to_skip)
        t_dialogues = MovieCorpus._multi_pop(t_dialogues, idx_to_skip)
        t_facts = MovieCorpus._multi_pop(t_facts, idx_to_skip)
        t_attitudes = MovieCorpus._multi_pop(t_attitudes, idx_to_skip)
        m_dialogues = MovieCorpus._multi_pop(m_dialogues, idx_to_skip)
        b_facts = MovieCorpus._multi_pop(b_facts, idx_to_skip)

        # --- cut into train and eval data ----------------------------------------------------------------------------
        cut_idx = int(len(x_attitudes) * 0.92)
        x_dialogues_train = x_dialogues[:cut_idx]
        t_dialogues_train = t_dialogues[:cut_idx]
        x_facts_train = x_facts[:cut_idx]
        t_facts_train = t_facts[:cut_idx]
        x_attitudes_train = x_attitudes[:cut_idx]
        t_attitudes_train = t_attitudes[:cut_idx]
        m_dialogues_train = m_dialogues[:cut_idx]
        b_facts_train = b_facts[:cut_idx]
        x_dialogues_eval = x_dialogues[cut_idx:]
        t_dialogues_eval = t_dialogues[cut_idx:]
        x_facts_eval = x_facts[cut_idx:]
        t_facts_eval = t_facts[cut_idx:]
        x_attitudes_eval = x_attitudes[cut_idx:]
        t_attitudes_eval = t_attitudes[cut_idx:]
        m_dialogues_eval = m_dialogues[cut_idx:]
        b_facts_eval = b_facts[cut_idx:]
        if self.params.head_type == "clf":
            x_wr_dialogues_train = x_wr_dialogues[:cut_idx]
            x_wr_dialogues_eval = x_wr_dialogues[cut_idx:]
        else:
            x_wr_dialogues_train = []
            x_wr_dialogues_eval = []

        # --- adapt to the tensorflow decoder model -------------------------------------------------------------------
        if self.params.head_type == "clf":
            x_train = []
            m_train = []
            y_train = []
            wr_idx_ = 0
            for i in range(2):
                wr_idx = [x + wr_idx_ for x in range(self.params.clf_pipes - 1)]
                x, m, y = self.generate_ndarray(x_dialogues_train, t_dialogues_train, x_facts_train,
                                                t_facts_train, x_attitudes_train, t_attitudes_train,
                                                m_dialogues_train, b_facts_train, self.params.n_ctx,
                                                x_wr_dialogues_train, wr_idx=wr_idx)
                x_train.append(x)
                m_train.append(m)
                y_train.append(y)
                wr_idx_ = wr_idx[-1] + 1
            wr_idx = [x for x in range(self.params.clf_pipes - 1)]
            x_eval, m_eval, y_eval = self.generate_ndarray(x_dialogues_eval, t_dialogues_eval, x_facts_eval,
                                                           t_facts_eval, x_attitudes_eval, t_attitudes_eval,
                                                           m_dialogues_eval, b_facts_eval, self.params.n_ctx,
                                                           x_wr_dialogues_eval, wr_idx=wr_idx)
        else:
            x_train, m_train, y_train = self.generate_ndarray(x_dialogues_train, t_dialogues_train, x_facts_train,
                                                              t_facts_train, x_attitudes_train, t_attitudes_train,
                                                              m_dialogues_train, b_facts_train, self.params.n_ctx)
            x_eval, m_eval, y_eval = self.generate_ndarray(x_dialogues_eval, t_dialogues_eval, x_facts_eval,
                                                           t_facts_eval, x_attitudes_eval, t_attitudes_eval,
                                                           m_dialogues_eval, b_facts_eval, self.params.n_ctx)

        return x_train, m_train, y_train, x_eval, m_eval, y_eval

    def generate_ndarray(self, x_dialogues, t_dialogues, x_facts, t_facts, x_attitudes,
                         t_attitudes, m_dialogues, b_facts, max_length, x_wr_dialogues=None, wr_idx=[0]):
        """ Generates one big array for masks and tokens. """
        num_of_samples = len(x_dialogues)
        x = np.zeros([num_of_samples, self.params.clf_pipes, max_length, 3], dtype=np.int32)
        m = np.zeros([num_of_samples, self.params.clf_pipes, max_length], dtype=np.int32)
        y = np.zeros([num_of_samples], dtype=np.int32)

        for i, (wd, td, wf, tf, wa, ta, m_, bf), in enumerate(zip(x_dialogues, t_dialogues, x_facts, t_facts,
                                                                  x_attitudes, t_attitudes, m_dialogues, b_facts)):

            # --- concatenate facts, attitudes and the dialogue both for the tokens (w) and the type-emb. (t) ---------
            x_wr = None
            if x_wr_dialogues is not None:
                wd = wd + [self.encoder['[classifier]']]
                td = td + [td[-1]]
                # x_wr = x_wr_dialogues[i][wr_idx] + [self.encoder['[classifier]']]
                x_wr = []
                for idx in wr_idx:
                    x_wr.append(x_wr_dialogues[i][idx] + [self.encoder['[classifier]']])
                m_ = m_ + [1]
            w_concat = wf + wa + wd
            t_concat = tf + ta + td

            # --- compute some sequence lengths -----------------------------------------------------------------------
            w_length = len(w_concat)
            fa_length = len(wf + wa)
            a_length = len(wa)
            d_length = len(wd)

            pos_emb_stt = self.params.n_vocab + self.params.n_special  # first id for the positional embedding

            yi = 0
            yi_ = 0

            # --- add the concatenated tokens and type-embeddings to the ndarray --------------------------------------
            if self.params.clf_pipes == 1:
                x[i, 0, :w_length, 0] = w_concat
                x[i, 0, :w_length, 1] = t_concat
                m[i, 0, fa_length:w_length] = m_
            else:
                yi = int(np.random.rand() * self.params.clf_pipes)
                y[i] = yi
                yi_s = [(yi + 1 + x) % self.params.clf_pipes for x in range(self.params.clf_pipes - 1)]
                w_concat_wr = [wf + wa + x_wr[idx] for idx in range(len(x_wr))]
                w_length_wr = [len(w_concat_wr[idx]) for idx in range(len(x_wr))]
                t_concat_wr = []
                for idx in range(len(x_wr)):
                    if len(wd) <= len(x_wr[idx]):
                        t_concat_wr.append(np.full([len(x_wr[idx])], td[-1]))
                        t_concat_wr[idx][:len(wd)] = td
                    else:
                        t_concat_wr.append(td[:len(x_wr[idx])])
                x[i, yi, :w_length, 0] = w_concat
                x[i, yi, :w_length, 1] = t_concat
                m[i, yi, fa_length:w_length] = m_
                for idx, yi_ in enumerate(yi_s):
                    x[i, yi_, :w_length_wr[idx], 0] = w_concat_wr[idx]
                    x[i, yi_, :w_length_wr[idx], 1] = np.concatenate((tf, ta, t_concat_wr[idx]), axis=0)

            # --- add positional embeddings ---------------------------------------------------------------------------
            # the order of facts and attitudes should not influence the result of our model. To ensure this,
            # we start with pos_emb_stt for every fact and attitude and iterate upon the next one.
            # starting with the facts:
            end = 0
            for b in bf:
                sst = end
                end = b
                x[i, :, sst:end, 2] = np.arange(pos_emb_stt, pos_emb_stt + end - sst)

            # continuing with the attitudes ...
            sst = end
            end = sst + a_length
            x[i, :, sst:end, 2] = np.full([a_length], pos_emb_stt)

            # and finally for the dialogue.
            # here we have to first compute the initial embedding
            if self.params.dynamic_pos_embeddings:
                poss_latest_start = 512 - d_length - 30  # TODO: Remove this hack
                pos_emb_stt_l = [np.random.randint(low=pos_emb_stt, high=pos_emb_stt + poss_latest_start)]
                if x_wr is not None:
                    for idx in range(len(x_wr)):
                        poss_latest_start = 512 - w_length_wr[idx] + 1  # TODO: Bug: w_length_wr is wrong length
                        pos_emb_stt_l.append(np.random.randint(low=pos_emb_stt, high=pos_emb_stt + poss_latest_start))
            else:
                if x_wr is None:
                    pos_emb_stt_l = [self.params.n_vocab + self.params.n_special]
                else:
                    pos_emb_stt_l = (len(x_wr) + 1) * [self.params.n_vocab + self.params.n_special]

            sst = end
            end = sst + d_length
            if self.params.clf_pipes == 1:
                x[i, 0, sst:end, 2] = np.arange(pos_emb_stt_l[0], pos_emb_stt_l[0] + d_length)
            elif self.params.clf_pipes == 999:
                end_wr = sst + len(x_wr)
                x[i, yi, sst:end, 2] = np.arange(pos_emb_stt_l[0], pos_emb_stt_l[0] + d_length)
                x[i, yi_, sst:end_wr, 2] = np.arange(pos_emb_stt_l[1], pos_emb_stt_l[1] + len(x_wr))
            else:
                end_wrs = [sst + len(x_wr[idx]) for idx in range(len(x_wr))]
                x[i, yi, sst:end, 2] = np.arange(pos_emb_stt_l[0], pos_emb_stt_l[0] + d_length)
                for idx, (end_wr, yi_) in enumerate(zip(end_wrs, yi_s)):
                    x[i, yi_, sst:end_wr, 2] = np.arange(pos_emb_stt_l[idx + 1],
                                                         pos_emb_stt_l[idx + 1] + len(x_wr[idx]))

        return x, m, y

    def process_attitudes(self, attitudes, speaker_turns, x, t, inference=False):
        """ Generates sequences for the attitudes of one dialogue """
        speaker = ['second_speaker', 'first_speaker']
        x_ = [[], []]
        t_ = [[], []]
        if inference:
            r = 1
        else:
            r = 2
        for i in range(r):
            bot_attitudes = attitudes[speaker[i % 2]]
            for attitude in bot_attitudes:
                if attitude['relation'] == "has_general_bot_attitude":
                    t_[i].extend([self.encoder[self.attitude_dict[attitude['subject']]]])
                    x_[i].extend([self.encoder["[att_{}]".format(attitude['object'])]])
                elif attitude['relation'] == "has_bot_certificate_attitude":
                    t_[i].extend([self.encoder[self.attitude_dict['certificate#0']]])
                    x_[i].extend([self.encoder["[att_{}]".format(attitude['object'])]])
                elif attitude['relation'] == "has_bot_budget_attitude":
                    t_[i].extend([self.encoder[self.attitude_dict['budget#0']]])
                    x_[i].extend([self.encoder["[att_{}]".format(attitude['object'])]])
        for i in range(speaker_turns):
            x.append(copy.deepcopy(x_[i % 2]))
            t.append(copy.deepcopy(t_[i % 2]))
        # for i in range(speaker_turns):
        #     x_ = []
        #     t_ = []
        #     bot_attitudes = attitudes[speaker[i % 2]]
        #     for attitude in bot_attitudes:
        #         if attitude['relation'] == "has_general_bot_attitude":
        #             t_.extend([self.encoder[self.attitude_dict[attitude['subject']]]])
        #             x_.extend([self.encoder["[att_{}]".format(attitude['object'])]])
        #         elif attitude['relation'] == "has_bot_certificate_attitude":
        #             t_.extend([self.encoder[self.attitude_dict['certificate#0']]])
        #             x_.extend([self.encoder["[att_{}]".format(attitude['object'])]])
        #         elif attitude['relation'] == "has_bot_budget_attitude":
        #             t_.extend([self.encoder[self.attitude_dict['budget#0']]])
        #             x_.extend([self.encoder["[att_{}]".format(attitude['object'])]])
        #     x.append(copy.deepcopy(x_))
        #     t.append(copy.deepcopy(t_))

    def process_facts(self, facts, speaker_turns, x, t, b, inference=False):
        """ Generates sequences for the facts of one dialogue """
        speaker = ['second_speaker', 'first_speaker']
        x_ = [[], []]
        t_ = [[], []]
        b_ = [[], []]
        if inference:
            r = 1
        else:
            r = 2
        for i in range(r):
            bot_facts = facts[speaker[i % 2]]
            for fact in bot_facts:
                tokenized_fact = self.text_encoder.encode([fact['object']])[0]
                l_fact = len(tokenized_fact)
                if fact['relation'] == "has_trivia":
                    t_[i].extend(l_fact * [self.encoder[self.fact_dict[fact['subject']]]])
                    x_[i].extend(tokenized_fact)
                    b_[i].append(len(x_[i]))
                elif fact['relation'] == "has_plot":
                    t_[i].extend(l_fact * [self.encoder[self.fact_dict['plot']]])
                    x_[i].extend(tokenized_fact)
                    b_[i].append(len(x_[i]))
        for i in range(speaker_turns):
            x.append(copy.deepcopy(x_[i % 2]))
            t.append(copy.deepcopy(t_[i % 2]))
            b.append(copy.deepcopy(b_[i % 2]))
        # for i in range(speaker_turns):
        #     x_ = []
        #     t_ = []
        #     b_ = []
        #     bot_facts = facts[speaker[i % 2]]
        #     for fact in bot_facts:
        #         tokenized_fact = self.text_encoder.encode([fact['object']])[0]
        #         l_fact = len(tokenized_fact)
        #         if fact['relation'] == "has_trivia":
        #             t_.extend(l_fact * [self.encoder[self.fact_dict[fact['subject']]]])
        #             x_.extend(tokenized_fact)
        #             b_.append(len(x_))
        #         elif fact['relation'] == "has_plot":
        #             t_.extend(l_fact * [self.encoder[self.fact_dict['plot']]])
        #             x_.extend(tokenized_fact)
        #             b_.append(len(x_))
        #     x.append(copy.deepcopy(x_))
        #     t.append(copy.deepcopy(t_))
        #     b.append(copy.deepcopy(b_))

    def generate_clf_wrong_utterances(self, dialogue, utterances, samples=1):
        """ Do something. """
        ref_story_type = dialogue['story'][2]['story_type']
        if ref_story_type == "PersonToMovieStory":
            ref_movie_title = dialogue['story'][0]['entities'][2]
        else:
            ref_movie_title = dialogue['story'][0]['entities'][0]
        ref_prepared_id = dialogue['story'][2]['prepared_id']

        wr_dialogues = []
        for wr_dialogue in self.dialogues_src:
            wr_story_type = wr_dialogue['story'][2]['story_type']
            if wr_story_type == "PersonToMovieStory":
                wr_movie_title = wr_dialogue['story'][0]['entities'][2]
            else:
                wr_movie_title = wr_dialogue['story'][0]['entities'][0]
            if wr_movie_title == ref_movie_title and ref_prepared_id != wr_dialogue['story'][2]['prepared_id']:
                wr_dialogues.append(copy.deepcopy(wr_dialogue))

        for wr_dialogue in wr_dialogues:
            wr_dialogue['dialogue_ner'] = self.process_dialogue(wr_dialogue['dialogue_ner'])

        def get_random():
            i = int(np.random.rand() * len(wr_dialogues))
            j = int(np.random.rand() * len(wr_dialogues[i]['dialogue_ner']))
            return wr_dialogues[i]['dialogue_ner'][j]

        outer = []
        for _ in range(utterances):
            inner = []
            for _ in range(samples):
                inner.append(get_random())
            outer.append(inner)

        return outer

    def process_dialogues(self, dialogue, x, t, m, compute_solo=False, x_wr=None, dialogue_src=None, num_x_wr=1):
        """ Generates sequences of one dialogue.

        This generates n samples for a dialogue with n speaker turns.
        The last utterance is always an utterance from the bot (means: has masks = 1 and bot-embedding-token).
        Therefore the algorithm decides between odd and even numbered utterances.

        Example:
            x    "how are you  ? " "fine  . "
            t     <h> <h> <h> <h>    <b> <b>
            m      0   0   0   0      1   1

            x    "how are you  ? " "fine  . " "great  ! "
            t     <b> <b> <b> <b>    <h> <h>    <b>  <b>
            m      1   1   1   1      0   0      1    1
            (with params.only_last_utterance_masking its possible to only have ones masking for the last utterance)

        Args:
            dialogue        A list of String. The dialogue in it's original form.
            x               A list. Object to store the processed sequences for word tokens.
            t               A list. Object to store the processed dialogue type embeddings.
            m               A list. Object to store the processed masks for the loss function.
            compute_solo    A boolean. If True, the minimum number of speaker turns is 0 (instead of 1).
            x_wr            A list. If not None, this algorithm additionally produces a wrong last utterance.
            dialogue_src    A dict. The original dialogue log file. Only needed for wrong utterances.
            num_x_wr        An integer. Number of wrong last utterances.
        """
        x_ = []      # dynamic growing dialogue
        x_wr_ = []   # dialogues with wrong last utterances
        t_even = []  # state embeddings for even dialogue
        t_odd = []
        m_even = []
        m_odd = []
        m_val = [
            self.encoder['[pos_human]'],
            self.encoder['[pos_bot]']
        ]

        # If True, the first utterance is computed as a solo utterance. This is only needed for inference.
        if compute_solo:
            incr = 1
            m_val = [
                # for inference, the last utterance is a user utterance
                self.encoder['[pos_bot]'],
                self.encoder['[pos_human]']
            ]
        else:
            incr = 0

        # for downward compatibility

        if x_wr is not None:
            x_wr_ = self.generate_clf_wrong_utterances(dialogue_src,
                                                       utterances=len(dialogue) - 1,
                                                       samples=(self.params.clf_pipes - 1) * 2)

        tokenized_dialogue = self.text_encoder.encode(dialogue)

        # --- utterances per sample are limited to boundaries (below) -------------------------------------------------
        if self.params.begin_cut_dialogues and not compute_solo:
            # compute number of utterances
            lower_boundary = 3
            upper_boundary = 7
            for i in range(len(tokenized_dialogue)):
                max_utt = i + 1  # maximum number of utterances at this point
                if max_utt > lower_boundary:
                    upper = np.min((upper_boundary, max_utt))
                    utterances = np.random.randint(low=lower_boundary, high=upper+1)
                elif i > 0:
                    utterances = i + 1
                else:
                    continue

                # iterate over dialogue and generate a sample
                x_ = []
                x_wr_temp_stack = []
                m_ = []
                t_ = []
                for j in range(utterances):
                    x_j = tokenized_dialogue[i + 1 - utterances + j]

                    # generate wrong dialogues
                    if (j >= utterances - 1) and x_wr is not None:
                        x_wr_temp_stack = []
                        for x_wr_utt in x_wr_[i - 1]:
                            x_wr_temp = copy.deepcopy(x_)
                            x_wr_temp += self.text_encoder.encode([x_wr_utt])[0]
                            x_wr_temp_stack.append(x_wr_temp)

                    # dialogue tokens
                    x_ += x_j

                    # state embedding tokens
                    t_ += (len(x_j) * [m_val[(utterances - j) % 2]])

                    # masks
                    if j < utterances - 1:
                        m_ += (len(x_j) * [0])
                    else:
                        m_ += (len(x_j) * [1])

                # add all to main list
                x.append(x_)
                m.append(m_)
                t.append(t_)
                x_wr.append(x_wr_temp_stack)

        # --- utterances per sample are limited to token size (e. g. 512) ---------------------------------------------
        else:
            for i, utterance in enumerate(tokenized_dialogue):
                l_utterance = len(utterance)
                x_prev = copy.deepcopy(x_)
                x_.extend(utterance)
                t_even.extend(l_utterance * [m_val[i % 2]])
                t_odd.extend(l_utterance * [m_val[(i + 1) % 2]])
                if self.params.only_last_utterance_masking:
                    m_even = list(np.multiply(m_even, 0))
                    m_odd = list(np.multiply(m_odd, 0))
                m_even.extend(l_utterance * [i % 2])
                m_odd.extend(l_utterance * [(i - 1) % 2])

                if i > (0 - incr):
                    x.append(copy.deepcopy(x_))
                    if x_wr is not None:
                        x_wr_add = []
                        for wr_utt in x_wr_[i - 1]:
                            wr_utt_tok = self.text_encoder.encode([wr_utt])[0]
                            x_wr_add.append(copy.deepcopy(x_prev + wr_utt_tok))
                        x_wr.append(x_wr_add)
                    if i % 2 == 0:
                        t.append(copy.deepcopy(t_odd))
                        m.append(copy.deepcopy(m_odd))
                    else:
                        t.append(copy.deepcopy(t_even))
                        m.append(copy.deepcopy(m_even))

    def int_to_txt(self, values):
        decoded_text = ""
        for x in values:
            decoded_text += str(self.text_encoder.decoder[x])
            decoded_text += " "
        decoded_text = decoded_text.replace("</w>", " ")
        decoded_text = decoded_text.replace("<unk>", "")
        return decoded_text

    @staticmethod
    def _multi_pop(list_content, list_idx):
        """ Returns a list of all elements from list_content without the indexes specified in list_idx. """
        return_list = []
        for idx, item in enumerate(list_content):
            if idx not in list_idx:
                return_list.append(item)
        return return_list

    def inference_preprocessing(self, dialogue, facts, attitudes):
        return ""

    def process_dialogue(self, dialogue):
        """ Replaces [eou] tokens and add [end] tokens.

        """
        new_dialogue = []
        for utterance in dialogue:
            tokens = utterance.split(" ")
            new_tokens = []
            for i in range(len(tokens)):
                if i == 0:
                    new_tokens.append(tokens[i])
                else:
                    if tokens[i] == "[eou]":
                        if tokens[i-1] in ["?", ".", ",", "!", ";", ":"]:
                            continue
                        else:
                            new_tokens.append(".")
                    else:
                        new_tokens.append(tokens[i])
            new_tokens.append("end#t")
            new_dialogue.append(" ".join(new_tokens))
        return new_dialogue


class Rocstories:
    def __init__(self, params):
        self.params = params
        self.text_encoder = text_utils.TextEncoder(params.encoder_path, params.bpe_path)
        self.encoder = self.text_encoder.encoder    # e.g.  'slogan</w>' = 36295
        self.params.n_vocab = len(self.text_encoder.encoder)
        self.params.n_special = 3       # i think for the three variables below (start, delim, classify)
        self.encoder['_start_'] = len(self.encoder)
        self.encoder['_delimiter_'] = len(self.encoder)
        self.encoder['_classify_'] = len(self.encoder)
        self.params.clf_token = self.encoder['_classify_']
        self.n_special = 3
        self.max_len = self.params.n_ctx // 2 - 2       # params.n_ctx = 512

    def prepare_rocstories(self):
        (trX1, trX2, trX3, trY), \
            (vaX1, vaX2, vaX3, vaY), \
            (teX1, teX2, teX3) = utils.encode_dataset(self.rocstories(self.params.data_dir),    # returns bpe of dataset
                                                      encoder=self.text_encoder)

        # Worst line of code, I saw in my life, ever!

        # Just computes the longest sequence in each input
        self.params.n_ctx = min(
            max(        # the '+' just concatenates the list together
                [len(x1[:self.max_len]) + max(len(x2[:self.max_len]), len(x3[:self.max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]
                + [len(x1[:self.max_len]) + max(len(x2[:self.max_len]), len(x3[:self.max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
                + [len(x1[:self.max_len]) + max(len(x2[:self.max_len]), len(x3[:self.max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)]
            ) + 3,      # why '+ 3' ?? maybe position encoding???
            self.params.n_ctx )
        # self.params.n_ctx = 77
        trX, trM = self.transform_roc(trX1, trX2, trX3)
        vaX, vaM = self.transform_roc(vaX1, vaX2, vaX3)

        if self.params.submit:      # False initially, since at train phase
            teX, teM = self.transform_roc(teX1, teX2, teX3)
        else:
            teX = None
            teM = None

        self.params.n_train = len(trY)
        self.params.n_valid = len(vaY)
        self.params.n_batch_train = self.params.n_batch * self.params.n_gpu     # determine how much is trained on the gpu before gradient update
        self.params.n_updates_total = (self.params.n_train // self.params.n_batch_train) * self.params.n_iter

        return trX, trM, trY, vaX, vaM, vaY, teX, teM

    def _rocstories(self, path):
        with open(path) as f:
            f = csv.reader(f)
            st = []
            ct1 = []
            ct2 = []
            y = []
            for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
                if i > 0:   # skip first line (heading of dataset)
                    s = ' '.join(line[1:5])
                    c1 = line[5]
                    c2 = line[6]
                    st.append(s)
                    ct1.append(c1)
                    ct2.append(c2)
                    y.append(int(line[-1])-1)
            return st, ct1, ct2, y

    def rocstories(self, data_dir, n_train=1497, n_valid=374):
        storys, comps1, comps2, ys = self._rocstories(os.path.join(data_dir,
                                                                   'cloze_test_val__spring2016 - '
                                                                   'cloze_test_ALL_val.csv'))
        teX1, teX2, teX3, _ = self._rocstories(os.path.join(data_dir,
                                                            'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
        tr_storys, va_storys, \
        tr_comps1, va_comps1, tr_comps2, va_comps2, \
        tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=SEED)

        trX1, trX2, trX3 = [], [], []
        trY = []
        for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
            trX1.append(s)
            trX2.append(c1)
            trX3.append(c2)
            trY.append(y)

        vaX1, vaX2, vaX3 = [], [], []
        vaY = []

        for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
            vaX1.append(s)
            vaX2.append(c1)
            vaX3.append(c2)
            vaY.append(y)

        trY = np.asarray(trY, dtype=np.int32)
        vaY = np.asarray(vaY, dtype=np.int32)
        return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

    def transform_roc(self, X1, X2, X3):
        n_batch = len(X1)       # = 1497
        xmb = np.zeros((n_batch, 2, self.params.n_ctx, 2), dtype=np.int32)  # first 2: x12, x13, second 2: labels
        mmb = np.zeros((n_batch, 2, self.params.n_ctx), dtype=np.float32)
        start = self.encoder['_start_']
        delimiter = self.encoder['_delimiter_']
        for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
            x12 = [start]+x1[:self.max_len]+[delimiter]+x2[:self.max_len]+[self.params.clf_token]   # clf_token = encoder['_classify_']
            x13 = [start]+x1[:self.max_len]+[delimiter]+x3[:self.max_len]+[self.params.clf_token]
            l12 = len(x12)
            l13 = len(x13)
            xmb[i, 0, :l12, 0] = x12    # so each input i (from 1 to 1497), contains a pair of right and wrong combinations. The labeling would be fixed after for loop
            xmb[i, 1, :l13, 0] = x13
            mmb[i, 0, :l12] = 1         # i guess a mask to know the positions???
            mmb[i, 1, :l13] = 1
        xmb[:, :, :, 1] = np.arange(self.params.n_vocab+self.params.n_special,
                                    self.params.n_vocab+self.params.n_special+self.params.n_ctx)    # fills up the last column with numbers from (40481 to 40481+77) (i guess for encoding the labels)
        return xmb, mmb


if __name__ == "__main__":

    class Params(dict):
        def __init__(self, *args, **kwargs):
            super(Params, self).__init__(*args, **kwargs)
            self.__dict__ = self


    params = Params()
    params.update({
        'bpe_path': "model/vocab_40000.bpe",
        'encoder_path': "model/encoder_bpe_40000.json",
        'data_dir': "data/",
        'n_vocab': 40000,
        'n_ctx': 512,
        'only_last_utterance_masking': True,
        'head_type': "clf",
        'clf_pipes': 3,
        'begin_cut_dialogues': False,
        'dynamic_pos_embeddings': True,
    })

    obj = MovieCorpus(params=params)
    obj.prepare_moviecorpus()


