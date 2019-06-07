""" Chatbot classes """
import sys
import copy
import imdb
import string
import pickle
import numpy as np

import tensorflow as tf

import model
import datasets
from include import beamsearch

from basic_chatbot import ChatBot

# --- settings --------------------------------------------------------------------------------------------------------
PATH_MOVIE_DATA = "data/moviecorpus/data.pkl"

MOVIES = ["Pulp Fiction", "Independence Day", "Casino Royale", "Lawrence of Arabia"]


class BasicChatbot(ChatBot):
    def __repr__(self):
        return "moviebot"

    def __init__(self, path_params, path_weights=None):
        super().__init__()
        with open(path_params, 'rb') as f:
            self.params = pickle.load(f)
        with open(PATH_MOVIE_DATA, 'rb') as f:
            self.movie_data = pickle.load(f)
        self.__change_parameter_for_inference()
        self.dialogue_histories = {}

        # TODO: Remove hack:
        self.params.desc = "moviecorpus"
        self.params.head_type = "clf"

        self.moviecorpus = datasets.MovieCorpus(self.params)
        self.pos_bot = self.moviecorpus.encoder['[pos_bot]']
        self.clf_token = self.moviecorpus.encoder['[classifier]']
        self.pos_emb_stt = self.params.n_vocab + self.params.n_special
        self.attitudes = {
            "Pulp Fiction": 4,
            "Bruce Willis": 1,
            "John Travolta": 2,
            "Uma Thurman": 5,
            "Quentin Tarantino": 3
        }
        self.model = model.TransformerDecoder(params=self.params)
        self.X = tf.placeholder(tf.int32, [1, None, 3])
        self.M = tf.placeholder(tf.float32, [1, None])
        self.Y = tf.placeholder(tf.int32, [None])
        _, self.lm_logits, _, _ = self.model.model(self.X,
                                                   self.M,
                                                   self.Y,
                                                   train=False,
                                                   reuse=tf.AUTO_REUSE,
                                                   greedy_decoding=True,
                                                   lm_logits_only=True)
        self.clf_logits, _, _, _ = self.model.model(self.X,
                                                    self.M,
                                                    self.Y,
                                                    train=False,
                                                    reuse=tf.AUTO_REUSE,
                                                    greedy_decoding=True)
        self.sess = tf.Session()
        self.model.load_checkpoint(self.sess, path=path_weights)

    def delete_histories(self, userids, delete_all=False):
        if delete_all:
            self.dialogue_histories = []
            return
        for userid in userids:
            if userid in self.dialogue_histories:
                self.dialogue_histories.pop(userid)
                return

    def generate_answer(self, utterance, userid, nlu_result=None):
        """ Computes named entities, generates input ndarray and decodes an answer. """

        # TODO: hack
        utterance += " end#t"
        movie = "pulp fiction"

        # check for existing dialogue
        if userid in self.dialogue_histories:
            dialogue_history = self.dialogue_histories[userid]
        else:
            dialogue_history = DialogHistory(userid=userid)
            self.dialogue_histories[userid] = dialogue_history

        # update dialogue history
        self._named_entity_resolution_v2(utterance, dialogue_history, nlu_result)
        self._gather_facts(dialogue_history)
        self._gather_attitudes(dialogue_history)

        # --- preprocessing for the dnn (TODO: exclude to a function) -------------------------------------------------
        x = []
        t = []
        m = []
        b = []

        self.moviecorpus.process_dialogues(dialogue=dialogue_history.dialogue_history, x=x, t=t, m=m, compute_solo=True)

        x = [x[-1]]
        t = [t[-1]]
        m = [m[-1]]

        self.moviecorpus.process_facts(facts={'second_speaker': dialogue_history.facts},
                                       speaker_turns=1, x=x, t=t, b=b, inference=True)
        self.moviecorpus.process_attitudes(attitudes={'second_speaker': dialogue_history.attitudes},
                                           speaker_turns=1, x=x, t=t, inference=True)

        X, M, _ = self.moviecorpus.generate_ndarray(x_dialogues=[x[0]], t_dialogues=[t[0]], x_facts=[x[1]],
                                                    t_facts=[t[1]], x_attitudes=[x[2]], t_attitudes=[t[2]],
                                                    m_dialogues=m,  b_facts=b,
                                                    max_length=len(x[0] + x[1] + x[2]))

        # --- decoding ------------------------------------------------------------------------------------------------
        X = np.reshape(X, [1, -1, 3])
        # ret = self._greedy_decoding(X=X)
        ret = self._beamsearch_decoding_v2(X=X)
        # ret = dialogue_history.dialogue_history[-1]
        print("RETURN: ")
        print(ret)
        print(" ")
        for key, value in self.moviecorpus.delex_dict.items():
            if value in ret:
                ret = ret.replace(value, key)
        dialogue_history.dialogue_history.append(ret)

        # lexicalization
        output = self._process_output(ret, dialogue_history)
        output = output.replace("do  n't", "don't")
        print("DIALOGUE HISTORY:")
        print(dialogue_history.dialogue_history)
        print(" ")

        return output, 0.5, "Acceptable"

    def _greedy_decoding(self, X):
        """ Generates a next answer """
        MAX_LENGTH = 20
        END_TOKEN = self.moviecorpus.encoder['[end]']
        # pad = np.array([[[0, 0, 0]]])
        x_length = X.shape[1]

        X_greedy = X
        for idx in range(MAX_LENGTH):
            # logits = self.sess.run(self.lm_logits, {self.X: np.concatenate((X_greedy, pad), axis=1)})
            logits = self.sess.run(self.lm_logits, {self.X: X_greedy})
            value = np.argmax(logits, 1)[-1]
            X_greedy = np.concatenate((X_greedy, np.array([[[value, self.pos_bot, X_greedy[0, -1, -1] + 1]]])),
                                      axis=1)
            if value == END_TOKEN:
                break

        result = X_greedy[0, x_length:, 0]
        result_txt = self.moviecorpus.int_to_txt(result)

        # giga hack (TODO: REMOVE AFTER NEW TRAINING)
        result_txt = result_txt.replace("[classifier]", "[end]")

        stop = "here"
        return result_txt

    def _beamsearch_decoding_v2(self, X):
        print("Beamsearch Decoding v2:")
        print("End Token ID: {}".format(self.moviecorpus.encoder['[end]']))
        beam_obj = beamsearch.BeamsearchDecoder(
            beam_size=4,
            max_length=20,
            end_token=self.moviecorpus.encoder['[end]'],
            alpha=0.6,
            min_beams=3,
            min_tokens=3,
            max_beams=10,
            pos_bot_token=self.pos_bot,
            clf_token=self.clf_token,
            model_in=self.X,
            model_out_lm=self.lm_logits,
            model_out_clf=self.clf_logits)

        result = beam_obj.beam_search(sess=self.sess, X=X)
        print("FINISHED:")
        for candidate in result:
            seq = self.moviecorpus.int_to_txt(candidate['seq'][0, -candidate['len']:, 0])
            print("Score: {}; {}. Text: {}".format(candidate['score'], candidate['clf_score'], seq))
        stop = "here"
        if len(result) < 1:
            return "i 'm sorry , my neurons could 'nt figure out a working sequence within my limitations ."
        return self.moviecorpus.int_to_txt(result[-1]['seq'][0, -result[-1]['len']:, 0])

    def _beamsearch_decoding(self, X, beam_size=4, max_length=20, end_token=None, alpha=0.6):
        """

        """
        if end_token is None:
            end_token = self.moviecorpus.encoder['[end]']

        def length_wu(length):
            return ((5 + length) ** alpha) / (5 + 1) ** alpha

        n_best_seqs = []

        logits = self.sess.run(self.lm_logits, {self.X: X})

        logits_last = logits[-1, :]

        logits_last_list = np.ndarray.tolist(logits_last)

        n_best = np.argpartition(logits_last, -4)[-4:]
        scores = logits_last[n_best]

        for val, score in zip(n_best, scores):
            n_best_seqs.append({
                'seq': np.concatenate((X, np.array([[[val, self.pos_bot, X[0, -1, -1] + 1]]])), axis=1),
                'score': np.log(score) / length_wu(1),
                'len': 1
            })

        n_best_seqs = sorted(n_best_seqs, key=lambda dic: dic['score'])

        finished = []

        for n_best in n_best_seqs:
            seq = self.moviecorpus.int_to_txt(n_best['seq'][0, -n_best['len']:, 0])
            print("Score: {}. Text: {}".format(n_best['score'], seq))

        for _ in range(max_length):
            new_candidates = []
            for seq in n_best_seqs:
                logits = self.sess.run(self.lm_logits, {self.X: seq['seq']})
                n_best = np.argpartition(logits[-1, :], -4)[-4:]
                scores = logits_last[n_best]
                for val, score in zip(n_best, scores):
                    candidate = {
                        'seq': np.concatenate((seq['seq'], np.array([[[val, self.pos_bot, seq['seq'][0, -1, -1] + 1]]])), axis=1),
                        'score': (seq['score'] + np.log(score)) / length_wu(seq['len'] + 1),
                        'len': seq['len'] + 1
                    }
                    if val == end_token:
                        finished.append(copy.deepcopy(candidate))
                    new_candidates.append(candidate)
            n_best_seqs = sorted(new_candidates, key=lambda dic: dic['score'])[-4:]

            # Finish beam search if an "end" token appears in the "best" beam.
            if n_best_seqs[-1]['seq'][0, -1, 0] == end_token:
                break

            for candidate in new_candidates:
                if candidate['seq'][0, -1, 0] == end_token:
                    candidate['score'] = -1e20

            n_best_seqs = sorted(new_candidates, key=lambda dic: dic['score'])[-4:]

            for n_best in n_best_seqs:
                seq = self.moviecorpus.int_to_txt(n_best['seq'][0, -n_best['len']:, 0])
                print("Score: {}. Text: {}".format(n_best['score'], seq))

        finished = sorted(finished, key=lambda dic: dic['score'])
        print("FINISHED:")
        for candidate in finished:
            seq = self.moviecorpus.int_to_txt(candidate['seq'][0, -candidate['len']:, 0])
            print("Score: {}. Text: {}".format(candidate['score'], seq))

        return self.moviecorpus.int_to_txt(finished[-1]['seq'][0, -finished[-1]['len']:, 0])

    def _named_entity_resolution_v2(self, utterance, dial_hist, nlu_result=None):
        """ New variant without a predefined movie title. """
        utterance = utterance.lower()
        detected_movie = None
        new_movie = False
        # check for nlu result

        # --- check for movie names ---
        for movie in MOVIES:
            if movie.lower() in utterance:
                detected_movie = movie
                if dial_hist.movie is None:
                    new_movie = True
                elif dial_hist.movie.title != detected_movie:
                    new_movie = True

                dial_hist.movie = self.__get_movie_data(movie)
                dial_hist.named_entities["movie#0"] = movie

                utterance = utterance.replace(movie.lower(), "movie#0")
                break

        # --- check for actor names ---
        for movie in MOVIES:
            movie_data = self.__get_movie_data(movie)
            actors = movie_data.get_fact("actors")
            for actor in actors:
                if str(actor).lower() in utterance:
                    if detected_movie is None:
                        detected_movie = movie
                        if dial_hist.movie is None:
                            new_movie = True
                        elif dial_hist.movie.title != detected_movie:
                            new_movie = True
                        dial_hist.movie = self.__get_movie_data(movie)

                    if movie == detected_movie:
                        dial_hist.named_entities["actor#0"] = str(actor)
                        utterance = utterance.replace(str(actor).lower(), "actor#0")
                    else:
                        dial_hist.named_entities["actor#1"] = str(actor)
                        utterance = utterance.replace(str(actor).lower(), "actor#1")

        # --- check for directors and writers (if no movie found) ---
        # TODO: Finish this
        for movie in MOVIES:
            movie_data = self.__get_movie_data(movie)
            # if str(movie_data.get_fact("director")).lower() in utterance:
            #     utterance = utterance.replace(str(movie_data.get_fact("director")).lower(), "director#0")
            #     dial_hist.named_entities["director#0"] = str(movie_data.get_fact("director"))
            #     dial_hist.movie = movie_data
            # if str(movie_data.get_fact("writer")).lower() in utterance:
            #     utterance = utterance.replace(str(movie_data.get_fact("writer")).lower(), "writer#0")
            #     dial_hist.named_entities["writer#0"] = str(movie_data.get_fact("writer"))
            #     dial_hist.movie = movie_data
            #     break

        # --- check nlu result (if no movie found) ---
        if not detected_movie and nlu_result is not None:
            nlu_result = ChatBot._get_nlu_intent(nlu_result=nlu_result, intent="ood-movie_generic")
            if nlu_result is not None:
                if nlu_result["entities"][0]["score"] > 0.5:
                    utterance = utterance.replace(nlu_result["entities"][0]["literal"], "movie#0")
                    detected_movie = string.capwords(nlu_result["entities"][0]["value"])
                    if dial_hist.movie is None:
                        new_movie = True
                    elif dial_hist.movie.title != detected_movie:
                        new_movie = True

                    dial_hist.movie = self.__get_movie_data(detected_movie)
                    dial_hist.named_entities["movie#0"] = dial_hist.movie.title

        # --- use a random movie (if no movie specified) ---
        if dial_hist.movie is None:
            random_movie = np.random.choice(MOVIES)
            dial_hist.movie = self.__get_movie_data(random_movie)
            dial_hist.named_entities["movie#0"] = random_movie

        # --- create or update possible entities ---
        if new_movie:
            if "actor#0" not in dial_hist.named_entities:
                dial_hist.named_entities["actor#0"] = str(np.random.choice(dial_hist.movie.get_fact("actors")))
            if "director#0" not in dial_hist.named_entities:
                dial_hist.named_entities["director#0"] = str(dial_hist.movie.get_fact("director"))
            if "writer#0" not in dial_hist.named_entities:
                dial_hist.named_entities["writer#0"] = str(dial_hist.movie.get_fact("writer"))

        # debug
        print("new movie: {}".format(new_movie))
        print("movie detected: {}".format(detected_movie))
        print("movie: {}".format(dial_hist.movie.title))

        dial_hist.dialogue_history.append(utterance)

    def _named_entity_resolution(self, utterance, dial_hist, movie=None):
        """ Detects all named entities, replaces them and additionally returns a dict. """

        if len(dial_hist.dialogue_history) > 0:
            # TODO: Check if useful
            if "movie#0" in dial_hist.named_entities:
                if movie is not None:
                    print("WARNING: Conversation has already a movie specified: '{}'.".format(movie))
        else:
            if movie is None:
                print("ERROR: Choosing automatically a movie is not implemented yet!")
                sys.exit()
            dial_hist.movie = self.__get_movie_data(movie)
            dial_hist.named_entities = {
                "movie#0": dial_hist.movie.title
            }
            dial_hist.possible_entities = {
                dial_hist.movie.title.lower(): "title",
            }
            for actor in dial_hist.movie.get_fact('actors'):
                dial_hist.possible_entities[str(actor)] = "actor"
            if dial_hist.movie.get_fact("writer=director"):
                dial_hist.possible_entities[str(dial_hist.movie.get_fact("director"))] = "writer,director"
            else:
                dial_hist.possible_entities[str(dial_hist.movie.get_fact("director"))] = "director"
                dial_hist.possible_entities[str(dial_hist.movie.get_fact("writer"))] = "writer"
            dial_hist.possible_entities[str(dial_hist.movie.get_fact("budget"))] = "budget"
            dial_hist.possible_entities[str(dial_hist.movie.get_fact("year"))] = "year"
            dial_hist.possible_entities[str(dial_hist.movie.get_fact("certificate"))] = "certificate"
            dial_hist.possible_entities[str(dial_hist.movie.get_fact("countries")[0])] = "country"
            dial_hist.possible_entities[str(dial_hist.movie.get_fact("genres")[0])] = "genre"

        # delexicalization with old named entities
        # (exact string match on lower case)
        for key, value in dial_hist.named_entities.items():
            utterance = utterance.replace(value.lower(), key)

        for key, value in dial_hist.possible_entities.items():
            if key.lower() in utterance:
                if value == "actor" and dial_hist.num_actors < 2:
                    dial_hist.named_entities["actor#{}".format(dial_hist.num_actors)] = key
                    dial_hist.num_actors += 1
                if "director" in value:
                    dial_hist.named_entities["director#0"] = key
                if "writer" in value:
                    dial_hist.named_entities["writer#0"] = key
                if value == "budget":
                    dial_hist.named_entities["budget#0"] = key
                if value == "year":
                    dial_hist.named_entities["year#0"] = key
                if value == "certificate":
                    dial_hist.named_entities["certificate#0"] = key
                if value == "genres":  # TODO: 2 genres should be possible!
                    dial_hist.named_entities["genre#0"] = key

        # delexicalization with updated named entities
        # (exact string match on lower case)
        for key, value in dial_hist.named_entities.items():
            utterance = utterance.replace(value.lower(), key)

        dial_hist.dialogue_history.append(utterance)

    def _gather_facts(self, dial_hist):

        if "movie#0" not in dial_hist.named_entities:
            raise Exception("Could not find movie in named_entities, but is required in this version.".format())

        # movie = self.__get_movie_data(movie=dial_hist.named_entities["movie#0"])
        movie = dial_hist.movie

        for key, value in dial_hist.named_entities.items():
            if "movie#" in key and not BasicChatbot.__subj_avail(dial_hist.facts, key):
                dial_hist.facts.append({
                    'subject': key,
                    'relation': "has_plot",
                    'object': movie.get_fact("plot")
                })
            if "actor#" in key and not BasicChatbot.__subj_avail(dial_hist.facts, key):
                if movie.has_trivia(key_entity=movie.title, entity=value) > 0:
                    dial_hist.facts.append(
                                {'subject': key,
                                 'relation': "has_trivia",
                                 'object': movie.get_trivia(key_entity=value, entity="None")})
                elif movie.has_trivia(key_entity=value, entity="None") > 0:
                    dial_hist.facts.append(
                                {'subject': key,
                                 'relation': "has_trivia",
                                 'object': movie.get_trivia(key_entity=value, entity="None")})
                dial_hist.facts.append(
                        {'subject': "movie#0",
                         'relation': "has_actor",
                         'object': key}
                )

    def _gather_attitudes(self, dial_hist):
        for key, value in dial_hist.named_entities.items():
            dial_hist.attitudes.append({
                'subject': key,
                'relation': "has_general_bot_attitude",
                'object': self._get_attitude(value)
            })

    def _get_attitude(self, entity):
        """ If no attitude specified, its marked as "unknown" (which means: 0) """
        if entity in self.attitudes:
            return self.attitudes[entity]
        else:
            return 0

    def _process_output(self, utterance, dial_hist):
        """ This function takes as input the output of a decoder and replaces placeholder with its correct tokens.

        """
        output_tokens = []
        utterance_tokens = utterance.split(" ")

        delex_dict_inverse = {v: k for k, v in self.moviecorpus.delex_dict.items()}

        for token in utterance_tokens:
            if token in self.moviecorpus.delex_dict:
                if token in dial_hist.named_entities:
                    output_token = dial_hist.named_entities[token]
                else:
                    if "budget" in token:
                        output_token = str(dial_hist.movie.get_fact("budget"))
                    elif "year" in token:
                        output_token = str(dial_hist.movie.get_fact("year"))
                    elif "certificate" in token:
                        output_token = str(dial_hist.movie.get_fact("certificate"))
                    elif "genre" in token:
                        output_token = str(dial_hist.movie.get_fact("genres")[0])
                    elif "country" in token:
                        output_token = str(dial_hist.movie.get_fact("countries")[0])
                    elif token == "end#t":
                        continue
                    elif "actor#" in token:
                        output_token = str(dial_hist.movie.get_fact("actors")[0])
                    elif "director#" in token:
                        output_token = str(dial_hist.movie.get_fact("director"))
                    elif "writer#" in token:
                        output_token = str(dial_hist.movie.get_fact("writer"))
                    else:
                        output_token = "unknown"

            else:
                output_token = token
            output_tokens.append(output_token)

        return " ".join(output_tokens)

    def __get_movie_data(self, movie):
        for movie_data in self.movie_data['movies']:
            if movie_data.title.lower() == movie.lower():
                return movie_data
        raise Exception(str(movie['movie'] + " not in self.movie_data. Should never happen!"))

    def __change_parameter_for_inference(self):
        """ Changes some parameter, that have to differ between training and inference.

        """
        self.params.clf_pipes = 1     # We don't need the classifier head in inference mode
        self.params.head_type = "clf"
        self.params.dynamic_pos_embeddings = False
        self.params.begin_cut_dialogues = False


    @staticmethod
    def __subj_avail(triple_dict, subject, relation=None, object=None):
        check_for_relation = False
        check_for_object = False
        if relation is not None:
            check_for_relation = True
        if object is not None:
            check_for_object = True
        for triple in triple_dict:
            if subject == triple['subject']:
                if check_for_relation and relation != triple['relation']:
                    continue
                if check_for_object and object != triple['object']:
                    continue
                return True
        return False


class DialogHistory:
    def __init__(self, userid):
        self.userid = userid
        self.movie = None
        self.dialogue_history = []
        self.named_entities = {}
        self.possible_entities = {}
        self.num_actors = 0
        self.facts = []
        self.attitudes = []


if __name__ == "__main__":
    chatbot = BasicChatbot(path_params="save/moviecorpus/moviecorpus_clf_pipes=2/params.pkl",
                           path_weights="save/moviecorpus/moviecorpus_clf_pipes=2/best_params.jl")
    chatbot.generate_answer(utterance="yesterday i have watched independence day . do you like john travolta ?", userid=1234)
    chatbot.generate_answer(utterance=" i like movies about gangster and stuff like that .", userid=1234)
