""" Class for beamsearch decoding

"""
import copy
import numpy as np


class BeamsearchDecoder:
    """

    """
    def __init__(self,
                 beam_size,         # An integer. Beam search size.
                 max_length,        # An integer. Maximum number of tokens per sequence.
                 end_token,         # An integer. The stop criteria for decoding.
                 alpha,             # A float. Sequence length normalization. Should be 0 < alpha < 1.
                 min_beams,         # An integer. Minimum number of finished beams before search ends.
                 min_tokens,        # An integer. Minimum number of tokens per finished sequence.
                 max_beams,         # An integer. Maximum number of finished beams after the search ends.
                 pos_bot_token,     # An integer. The content encoding token for a bot utterance.
                 clf_token,         # An integer. The classifier token.
                 model_in,          # A tensorflow node. The node where the input sequence can be stored.
                 model_out_lm,      # A tensorflow node. The language model logits node of the tensorflow graph.
                 model_out_clf):    # A tensorflow node. The classifier logits node of the tensorflow graph.
        self.beam_size = beam_size
        self.max_length = max_length
        self.end_token = end_token
        self.alpha = alpha
        self.min_beams = min_beams
        self.min_tokens = min_tokens
        self.max_beams = max_beams
        self.pos_bot_token = pos_bot_token
        self.clf_token = clf_token
        self.model_in = model_in
        self.model_out_lm = model_out_lm
        self.model_out_clf = model_out_clf

        self.candidates = []
        self.n_best_candidates = []
        self.finished = []

    def beam_search(self, sess, X):
        """ Beamsearch algorithm

        Params:
            sess    A tf.Session object. The tensorflow session which should be used for token prediction.
            X       A list of integer. The start sequence.
            x_in    A tensorflow node. The node where the input sequence can be stored.
            x_out   A tensorflow node. The logits node of the tensorflow graph.

        """
        self.candidates = []
        self.finished = []
        self.n_best_candidates = [{
            'seq': X,
            'score': 0,
            'len': 0
        }]

        for step in range(self.max_length):
            # compute all candidates
            candidates = self._candidate_search(sess=sess, n_best_seq=self.n_best_candidates)

            # continue beam search, if minimum token length is not reached
            if step < self.min_tokens:
                for candidate in candidates:
                    if candidate['seq'][0, -1, 0] == self.end_token:
                        candidate['score'] = -1e20
                self.n_best_candidates = sorted(candidates, key=lambda dic: dic['score'])[-self.beam_size:]
                continue

            # save all finished candidates
            for candidate in candidates:
                if candidate['seq'][0, -1, 0] == self.end_token:
                    self.finished.append(copy.deepcopy(candidate))

            # check for stop condition: best candidate is a finished sequence
            self.n_best_candidates = sorted(candidates, key=lambda dic: dic['score'])[-self.beam_size:]
            if self.n_best_candidates[-1]['seq'][0, -1, 0] == self.end_token \
                    and len(self.n_best_candidates) >= self.min_beams:
                break

            # check for stop condition: maximum number of beams reached.
            if len(self.n_best_candidates) >= self.max_beams:
                break

            for candidate in candidates:
                if candidate['seq'][0, -1, 0] == self.end_token:
                    candidate['score'] = -1e20
                if self._check_for_double_n_grams(candidate):
                    candidate['score'] = -1e20
            self.n_best_candidates = sorted(candidates, key=lambda dic: dic['score'])[-self.beam_size:]

        finished = sorted(self.finished, key=lambda dic: dic['score'])

        self._clf_score(candidates=finished, sess=sess)

        return finished

    def _predict_next_token(self, sess, X):
        """ Computes the most relevant next tokens. """
        logits = sess.run(self.model_out_lm, {self.model_in: X})
        logits_last = logits[-1, :]
        n_best_logits = np.argpartition(logits_last, -self.beam_size)[-self.beam_size:]
        n_best_scores = logits_last[n_best_logits]

        return n_best_logits, n_best_scores

    def _candidate_search(self, sess, n_best_seq):
        """ Computes the most relevant sequence candidates """
        candidates = []
        for seq in n_best_seq:
            logits, scores = self._predict_next_token(sess=sess, X=seq['seq'])
            for val, score in zip(logits, scores):
                candidates.append({
                    'seq': np.concatenate((seq['seq'], np.array([[[val, self.pos_bot_token, seq['seq'][0, -1, -1] + 1]]])),
                                          axis=1),
                    'score': (seq['score'] + np.log(score)) / self._length_wu(seq['len'] + 1),
                    'len': seq['len'] + 1
                })

        return candidates

    def _clf_score(self, candidates, sess):
        """ Computes the classification score of finished sequences. """
        for candidate in candidates:
            # compute classifier loss
            model_in = np.concatenate((candidate['seq'], np.array([[[self.clf_token,
                                                                     self.pos_bot_token,
                                                                     candidate['seq'][0, -1, -1] + 1]]])),
                                      axis=1)
            clf_logits = sess.run(self.model_out_clf, {self.model_in: model_in})
            stop = "here"
            candidate['clf_score'] = clf_logits

    def _length_wu(self, length):
        return ((5 + length) ** self.alpha) / (5 + 1) ** self.alpha

    def _check_for_double_n_grams(self, candidate, n=3):
        if candidate['len'] >= 2*n:
            n_gram = candidate['seq'][:, -n:, 0]
            for step in range(candidate['len'] - 2*n + 1):
                sst = -candidate['len'] + step
                end = -candidate['len'] + step + n
                if np.array_equal(candidate['seq'][:, sst:end, 0], n_gram):
                    return True
        return False
