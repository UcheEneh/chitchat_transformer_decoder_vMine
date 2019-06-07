# --- pythonpath hack ---
import sys
sys.path.insert(0, "./soccer_chatbot")
sys.path.insert(0, "./soccer_chatbot/models")

from models.kg_copy_model import KGSentient
from args import get_args
import numpy as np
import os, traceback
from collections import defaultdict
#from batcher import DialogBatcher
from utils_new import load_model
#from flask import Flask, request
from BotAgent import BotAgent
import torch
from abc import ABC, abstractmethod

#get arguments
args = get_args()

from basic_chatbot import ChatBot


class SoccerBot(ChatBot):
    def __init__(self):
        #userdictionary
        super().__init__()
        self.dialogue_hist = defaultdict(list)
        # Set random seed
        np.random.seed(args.randseed)
        torch.manual_seed(args.randseed)

        if args.gpu:
            torch.cuda.manual_seed(args.randseed)

        if os.path.isfile(args.stoi):
            self.stoi = np.load(args.stoi, allow_pickle=True).item()

        self.itos = {v: k for k, v in self.stoi.items()}
        # Get data
        # chat_data = DialogBatcher(gpu=args.gpu)
        self.model = KGSentient(hidden_size=args.hidden_size, max_r=args.resp_len, gpu=args.gpu, n_words=len(self.stoi) + 1,
                           emb_dim=args.words_dim, kb_max_size=200, b_size=args.batch_size, lr=args.lr,
                           dropout=args.rnn_dropout, emb_drop=args.emb_drop, teacher_forcing_ratio=args.teacher_forcing,
                           sos_tok=self.stoi['<sos>'], eos_tok=self.stoi['<eos>'],
                           itos=self.itos, first_kg_token=self.stoi['o0'])
        if args.gpu:
            self.model = self.model.cuda()

        self.model_name = 'Sentient_model2'
        self.model = load_model(self.model, self.model_name, gpu=args.gpu)
        self.bot = BotAgent(self.model, stoi=self.stoi, itos=self.itos)

    def generate_answer(self, utterance, userid, nlu_result=None):
        team = 'Argentina_kg'
        try:
            if not self.dialogue_hist[userid]:
                #query = request.form['text']
                #rint (query)
                self.dialogue_hist[userid].append(utterance)
                resp = self.bot.response(utterance, team)
                #resp = jsonify({"status": "success","response":resp[0]})
                #if not request.cookies.get('query'):
                #resp.set_cookie('query', request.cookies.get('query') + ' '+query)
                #else:
                #    resp.set_cookie('query', request.cookies.get('query') + ' ' + query)
                #print (request.cookies.get('query'))
                self.dialogue_hist[userid].append(resp[0])
                return resp[0], 0.5, "Acceptable"
            else:
                self.dialogue_hist[userid].append(utterance)
                query = ' '.join(q for q in self.dialogue_hist[userid])
                resp = self.bot.response(query, team)
                self.dialogue_hist[userid].append(resp[0])
                print (resp[0])
                return resp[0], 0.5, "Acceptable"
            #print (self.dialogue_hist[userid])

        except Exception as e:
            #print (e)
            traceback.print_exc(file=sys.stdout)
            return "", 0, "Acceptable"

    def delete_histories(self, userids, delete_all=False):
        for userid in userids:
            if userid in self.dialogue_hist:
                self.dialogue_hist.pop(userid)
            else:
                print("Soccerbot warning: Userid not found in dict.")
        if delete_all:
            self.dialogue_hist = defaultdict(list)

    def __repr__(self):
        return("soccerbot")


if __name__ == '__main__':
    mybot = SoccerBot()
    mybot.generate_answer('i like the team pretty much', 'z12', 'Argentina_kg')
    mybot.generate_answer('who is the captain of argentina?', 'z12', 'Argentina_kg')
    mybot.generate_answer('do you know the name of their coach?', 'z12', 'Argentina_kg')