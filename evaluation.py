""" Evaluation Functions

"""
import sys
import csv
import copy
import glob
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk import bleu

import model
import datasets
from include import beamsearch

# --- settings --------------------------------------------------------------------------------------------------------
EXPERIMENT_NAME = ""
PATH_PARAMS = "save/moviecorpus/moviecorpus_clf_pipes=3_new/params.pkl"
PATH_WEIGHTS = "save/moviecorpus/moviecorpus_clf_pipes=3_new/best_params.jl"

PATH_TEST_DATA_RAW = "data/moviecorpus/some_goods/*"
DATASET_EVALUATION = False
CALCUATE_BLEU = True

PATH_CSV_FILE = "some_goods_2_clf3_{}.csv"
NUMBER_OF_ROWS_PER_FILE = 100
CSV_KEYS = ['movie', 'utt_1', 'utt_2', 'utt_3', 'utt_4', 'utt_5', 'utt_6', 'entity_1', 'entity_2', 'entity_3',
            'entity_4', 'value_1', 'value_2', 'value_3', 'value_4', 'fact_1', 'fact_2', 'fact_3', 'fact_4', 'fact_5']


# --- evaluation class ------------------------------------------------------------------------------------------------
class Evaluation:
    def __init__(self, path_params, path_weights):
        with open(path_params, 'rb') as f:
            self.params = pickle.load(f)

        if not hasattr(self.params, 'dynamic_pos_embeddings'):
            self.params.dynamic_pos_embeddings = False
        if not hasattr(self.params, 'begin_cut_dialogues'):
            self.params.begin_cut_dialogues = False
        if not hasattr(self.params, 'only_last_utterance_masking'):
            self.params.only_last_utterance_masking = False

        self.__change_parameter_for_inference()

        self.moviecorpus = datasets.MovieCorpus(self.params)
        self.begin_num = 0

        # --- load test data ---
        logfiles = glob.glob(PATH_TEST_DATA_RAW)
        self.dialogues = []
        for logfile in logfiles:
            with open(logfile, 'r', encoding='utf-8') as f:
                self.dialogues.append(json.load(f))

        # --- init and load model ---
        if not DATASET_EVALUATION:
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

            # --- beam search ---
            self.beam_obj = beamsearch.BeamsearchDecoder(
                beam_size=4,
                max_length=20,
                end_token=self.moviecorpus.encoder['[end]'],
                alpha=0.6,
                min_beams=3,
                min_tokens=3,
                max_beams=10,
                pos_bot_token=self.moviecorpus.encoder['[pos_bot]'],
                clf_token=self.moviecorpus.encoder['[classifier]'],
                model_in=self.X,
                model_out_lm=self.lm_logits,
                model_out_clf=self.clf_logits)

        self.attitude_dict = {
            0: "I don't know.",
            1: "I really don't like.",
            2: "I don't like.",
            3: "I like.",
            4: "I like very much.",
            5: "My favourite.",
            6: "No opinion.",  # 0 (age)
            8: "Disagree with it.",  # 2 (age)
            9: "Agree with it."  # 3(age)
        }

    def generate_cross_entropy(self):
        """ Computes cross entropy """
        for dialogue_src in self.dialogues:
            x_dialogues = []
            t_dialogues = []
            m_dialogues = []
            x_facts = []
            t_facts = []
            b_facts = []
            x_attitudes = []
            t_attitudes = []
            dialogue = dialogue_src['dialogue_ner']
            dialogue = self.moviecorpus.process_dialogue(dialogue=dialogue)
            facts = dialogue_src['facts']
            attitudes = dialogue_src['attitudes']
            speaker_turns = len(dialogue) - 1
            self.moviecorpus.process_dialogues(dialogue=dialogue, x=x_dialogues, t=t_dialogues, m=m_dialogues, compute_solo=True)
            self.moviecorpus.process_facts(facts=facts, speaker_turns=speaker_turns, x=x_facts, t=t_facts, b=b_facts)
            self.moviecorpus.process_attitudes(attitudes=attitudes, speaker_turns=speaker_turns, x=x_attitudes, t=t_attitudes)

    def generate_csv_for_human_evaluation_dataset_evaluation(self):
        """ Goes through all the test logfiles and generates a csv file.

        """
        self.begin_num = 0
        with open(PATH_CSV_FILE.format(self.begin_num), 'w', newline='') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(CSV_KEYS)

        keys = ['second_speaker', 'first_speaker']
        cnt = 0

        for dialogue_src in self.dialogues:
            dialogue = dialogue_src['dialogue_ner']
            if len(dialogue) < 6:
                continue
            dialogue = self.moviecorpus.process_dialogue(dialogue=dialogue)
            speaker_turns = len(dialogue) - 1

            facts = self.__generate_facts(facts_original=dialogue_src['facts_original'], keys=keys)
            attitudes = self.__generate_attitudes(attitudes_original=dialogue_src['attitudes_original'], keys=keys)

            if dialogue_src['story'][2]['story_type'] == "PersonToMovieStory":
                movie_title = dialogue_src['story'][0]['entities'][2]
            else:
                movie_title = dialogue_src['story'][0]['entities'][0]

            for i in range(speaker_turns):
                utterances = []
                for j in range(6):
                    if i - 4 + j >= 0:
                        utterances.append(self.__restore_named_entities(dialogue[i - 4 + j],
                                                                        dialogue_src['named_entity_dict']))
                if True:  # np.random.rand() > 0.5:
                    cnt += 1
                    self.__write_csv_line(utts=utterances,
                                          entities=attitudes[keys[i % 2]]['entities'],
                                          values=attitudes[keys[i % 2]]['values'],
                                          facts=facts[keys[i % 2]],
                                          movie_title=movie_title)
                    if cnt == NUMBER_OF_ROWS_PER_FILE:
                        cnt = 0
                        self.begin_num += 1
                        with open(PATH_CSV_FILE.format(self.begin_num), 'w', newline='', encoding='utf-8') as f:
                            csv_file = csv.writer(f)
                            csv_file.writerow(CSV_KEYS)

    def generate_csv_for_human_evaluation(self):
        """ Goes through all the test logfiles and generates a csv file.

        """

        with open(PATH_CSV_FILE.format(self.begin_num), 'w', newline='', encoding='utf-8') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(CSV_KEYS)

        cnt = 0

        for dialogue_src in self.dialogues:
            x_dialogues = []
            t_dialogues = []
            m_dialogues = []
            x_facts = []
            t_facts = []
            b_facts = []
            x_attitudes = []
            t_attitudes = []
            dialogue = dialogue_src['dialogue_ner']
            dialogue = self.moviecorpus.process_dialogue(dialogue=dialogue)
            facts = dialogue_src['facts']
            attitudes = dialogue_src['attitudes']
            speaker_turns = len(dialogue) - 1
            self.moviecorpus.process_dialogues(dialogue=dialogue, x=x_dialogues, t=t_dialogues, m=m_dialogues, compute_solo=True)
            self.moviecorpus.process_facts(facts=facts, speaker_turns=speaker_turns, x=x_facts, t=t_facts, b=b_facts)
            self.moviecorpus.process_attitudes(attitudes=attitudes, speaker_turns=speaker_turns, x=x_attitudes, t=t_attitudes)

            if dialogue_src['story'][2]['story_type'] == "PersonToMovieStory":
                movie_title = dialogue_src['story'][0]['entities'][2]
            else:
                movie_title = dialogue_src['story'][0]['entities'][0]

            facts_processed = []
            keys = ['second_speaker', 'first_speaker']

            facts_csv = self.__generate_facts(facts_original=dialogue_src['facts_original'], keys=keys)
            attitudes_csv = self.__generate_attitudes(attitudes_original=dialogue_src['attitudes_original'], keys=keys)

            # generate all answers with beamsearch
            for i in range(speaker_turns):
                X, M, _ = self.moviecorpus.generate_ndarray(x_dialogues=[x_dialogues[i]], t_dialogues=[t_dialogues[i]],
                                                            x_facts=[x_facts[i]],
                                                            t_facts=[t_facts[i]], x_attitudes=[x_attitudes[i]],
                                                            t_attitudes=[t_attitudes[i]],
                                                            m_dialogues=[m_dialogues[i]], b_facts=[b_facts[i]],
                                                            max_length=len(x_dialogues[i] + x_facts[i] + x_attitudes[i]))
                result = self.beam_obj.beam_search(self.sess, X=np.reshape(X, [1, -1, 3]))
                print(" NEW UTTERANCE ")
                for candidate in result:
                    seq = self.moviecorpus.int_to_txt(candidate['seq'][0, -candidate['len']:, 0])
                    print("Score: {}; {}. Text: {}".format(candidate['score'], candidate['clf_score'], seq))
                    candidate['txt'] = seq
                    candidate['final_score'] = candidate['score'] + candidate['clf_score'][0]
                result = sorted(result, key=lambda dic: dic['final_score'])
                if len(result) < 1:
                    continue

                # compute bleu


                best_utterance = result[-1]['txt']
                utterances = []
                for j in range(5):
                    if i - 4 + j >= 0:
                        utterances.append(self.__restore_named_entities(dialogue[i - 4 + j],
                                                                        dialogue_src['named_entity_dict']))

                for key, value in self.moviecorpus.delex_dict.items():
                    best_utterance = best_utterance.replace(value, key)

                utterances.append(self.__restore_named_entities(best_utterance,
                                                                dialogue_src['named_entity_dict']))

                if True:  # np.random.rand() > 0.5:
                    cnt += 1
                    self.__write_csv_line(utts=utterances,
                                          entities=attitudes_csv[keys[i % 2]]['entities'],
                                          values=attitudes_csv[keys[i % 2]]['values'],
                                          facts=facts_csv[keys[i % 2]],
                                          movie_title=movie_title)
                    if cnt >= NUMBER_OF_ROWS_PER_FILE:
                        cnt = 0
                        self.begin_num += 1
                        with open(PATH_CSV_FILE.format(self.begin_num), 'w', newline='', encoding='utf-8') as f:
                            csv_file = csv.writer(f)
                            csv_file.writerow(CSV_KEYS)
                stop = "here"

    def __change_parameter_for_inference(self):
        """ Changes some parameter, that have to differ between training and inference.

        """
        self.params.clf_pipes = 1     # We don't need the classifier head in inference mode
        self.params.head_type = "clf"
        self.params.dynamic_pos_embeddings = False
        self.params.begin_cut_dialogues = False

    def __write_csv_line(self, utts, entities, values, facts, movie_title):
        """
        """
        facts_csv = copy.deepcopy(facts)
        entities_csv = copy.deepcopy(entities)
        values_csv = copy.deepcopy(values)
        utts_csv = copy.deepcopy(utts)

        for i in range(len(values_csv)):
            values_csv[i] = self.attitude_dict[values_csv[i]]
        for i in range(len(utts_csv)):
            utts_csv[i] = utts_csv[i].replace("end#t", "")
            utts_csv[i] = utts_csv[i].replace(",", "&#44")
        while len(entities_csv) < 4:
            entities_csv.append(" ")
        while len(values_csv) < 4:
            values_csv.append(" ")
        while len(utts_csv) < 6:
            utts_csv.insert(0, " ")
        while len(facts_csv) < 5:
            facts_csv.append(" ")
        with open(PATH_CSV_FILE.format(self.begin_num), 'a', newline='', encoding='utf-8') as f:
            csv_file = csv.writer(f)
            csv_file.writerow([movie_title] + utts_csv + entities_csv + values_csv + facts_csv)

    def __restore_named_entities(self, utterance, named_entity_dict):
        for key, value in named_entity_dict.items():
            utterance = utterance.replace(value, key)
        return utterance

    def __generate_attitudes(self, attitudes_original, keys):
        attitudes = {
            'first_speaker': {},
            'second_speaker': {}
        }
        for i in range(2):
            attitudes[keys[i]]['entities'] = []
            attitudes[keys[i]]['values'] = []
            for item in attitudes_original[keys[i]]:
                if item['relation'] == "has_general_bot_attitude":
                    attitudes[keys[i]]['entities'].append(item['subject'])
                    attitudes[keys[i]]['values'].append(item['object'])
                elif item['relation'] == "has_bot_certificate_attitude":
                    attitudes[keys[i]]['entities'].append("Age restriction")
                    attitudes[keys[i]]['values'].append(item['object'] + 6)
                else:
                    raise ValueError("'{}' not handled.".format(item['relation']))

        return attitudes

    def __generate_facts(self, facts_original, keys):
        facts = {
            'first_speaker': [],
            'second_speaker': []
        }
        for i in range(2):
            for item in facts_original[keys[i]]:
                if item['relation'] == "has_release_year":
                    facts[keys[i]].append("Release year: {}".format(item['object']))
                elif item['relation'] == "has_actor":
                    facts[keys[i]].append("Actor: {}".format(item['object']))
                elif item['relation'] == "has_director":
                    facts[keys[i]].append("Director: {}".format(item['object']))
                elif item['relation'] == "has_writer":
                    facts[keys[i]].append("Writer: {}".format(item['object']))
                elif item['relation'] == "has_genre":
                    facts[keys[i]].append("Genre: {}".format(item['object']))
                elif item['relation'] == "has_shot_location":
                    facts[keys[i]].append("Shot location: {}".format(item['object']))
                elif item['relation'] == "has_budget":
                    facts[keys[i]].append("Budget: {}".format(item['object']))
                elif item['relation'] == "has_age_certificate":
                    facts[keys[i]].append("Age restriction: {} years.".format(item['object']))
                elif item['relation'] in ['has_plot', 'has_trivia']:
                    continue
                else:
                    raise KeyError("'{}' not handled.".format(item['relation']))
        return facts


if __name__ == "__main__":
    eval_obj = Evaluation(path_params=PATH_PARAMS, path_weights=PATH_WEIGHTS)
    if DATASET_EVALUATION:
        eval_obj.generate_csv_for_human_evaluation_dataset_evaluation()
    else:
        eval_obj.generate_csv_for_human_evaluation()
