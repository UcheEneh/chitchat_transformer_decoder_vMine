import re
import ftfy
import json
import spacy

from tqdm import tqdm

# returns the word as a combination of a pair of letters in it
def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs    # e.g. 'decided' returns: <class 'set'>: {('d', 'e'), ('e', 'c'), ('c', 'i'), ('i', 'd'), ('e', 'd</w>'),}


def text_standardize(text, strip=True):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    # text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|}+|{+|\|+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    if strip:
        return text.strip()
    else:
        return text


class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path, delex_dict=None):
        self.delex_dict = delex_dict
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', 'textcat'])  # old: en
        with open(encoder_path) as f:
            self.encoder = json.load(f)     # encoder dict contains word to embedding (encode_bpe_40000)
        self.decoder = None
        self.update_decoder()
        with open(bpe_path) as f:
            merges = f.read().split('\n')[1:-1]     # [1:-1] everything from (and including) the second element, up to
                                            # (but not including) the last element (?? it does include last element).
        merges = [tuple(merge.split()) for merge in merges]     # created as a tuple so it can be stored in dict below
        self.bpe_ranks = dict(zip(merges, range(len(merges))))  # e.g:   self.bpe_ranks[('comp', 'el</w>')] = 27648
        self.cache = {}

    def update_decoder(self):
        self.decoder = {v:k for k,v in self.encoder.items()}    # decoder converts from embedding to word
                                                                # inverse of encoder

    # creating bit-pair encoding
    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)   # creates a tuple (basically a list of all the letters in the token(word)), and adds </w> to the last letter
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)     # e.g. 'decided' returns: <class 'set'>: {('d', 'e'), ('e', 'c'), ('c', 'i'), ('i', 'd'), ('e', 'd</w>'),}

        if not pairs:   # in single letter case
            return token+'</w>'

        while True:     # key below: - key function where each argument from pairs is passed, and comparison is performed based on its return value
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))      # if the pair e.g. ('d', 'e') is in the bpe_ranks dict, it returns the value, else a float('inf') value. It returns the min of all the pairs
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)    # 'i' just tells us where to start looking for the letter in variable 'first'
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    # takes in text and returns the bpe encoding
    def encode(self, texts, verbose=True, tokenize=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False, position=1):
                if tokenize:
                    # ftfy.fix_text: fixes unicode e.g. fixes schöne error: schÃ¶n
                    text = self.nlp(text_standardize(ftfy.fix_text(text)))
                else:
                    text = text_standardize(text)
                text_tokens = []
                for token in text:
                    if self.delex_dict is not None:     # None for Rocstories
                        if str(token) in self.delex_dict:
                            text_tokens.append(self.encoder[self.delex_dict[str(token)]])
                            continue
                    # self.bpe(): takes in a word, if it's a single letter, returns the letter + </w> (# e.g. for only 'I': i</w>, encoder[i</w>] = 249 is then added to the list text_tokens ... and so on)
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens
