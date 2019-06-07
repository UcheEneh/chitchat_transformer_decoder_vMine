import numpy as np
import text_utils


class Dictionary:
    def __init__(self, params):
        self.params = params
        self.text_encoder = text_utils.TextEncoder(params.encoder_path, params.bpe_path)
        self.encoder = self.text_encoder.encoder
        self.params.n_vocab = len(self.text_encoder.encoder)
        self.params.n_special = 3
        self.encoder['_start_'] = len(self.encoder)
        self.encoder['_delimiter_'] = len(self.encoder)
        self.encoder['_classify_'] = len(self.encoder)
        self.params.clf_token = self.encoder['_classify_']

    def txt_to_int(self, txt):
        return self.text_encoder.encode([txt])

    def int_to_txt(self, values):
        decoded_text = ""
        for x in values:
            decoded_text += str(self.text_encoder.decoder[x])
            decoded_text += " "
        # decoded_text = decoded_text.replace("</w>", " ")
        # decoded_text = decoded_text.replace("<unk>", "")
        return decoded_text

    def transform(self, x):
        transformed_x = np.zeros((len(x), 1, self.params.n_ctx, 2), dtype=np.int32)
        transformed_m = np.zeros((len(x), 1, self.params.n_ctx), dtype=np.int32)

        for i, item in enumerate(x):
            l_x = len(item)
            transformed_x[i, 0, :l_x, 0] = item
            transformed_m[i, 0, :l_x] = 1
        transformed_x[:, :, :, 1] = np.arange(self.params.n_vocab+self.params.n_special,
                                              self.params.n_vocab+self.params.n_special+self.params.n_ctx)

        return transformed_x, transformed_m

    def transform_v2(self, x, add=0):
        if type(x) is not list:
            x = np.ndarray.tolist(x)
        pos_emb = np.arange(self.params.n_vocab + self.params.n_special,
                            self.params.n_vocab + self.params.n_special + len(x) + add)
        stacked = np.stack(((x + add * [0]), pos_emb), 1)
        stacked = np.expand_dims(stacked, 0)
        return stacked
