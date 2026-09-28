import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [x for x in result if x.strip() != '']
        encoded_list = [self.vocab[token] if token in self.vocab else self.vocab['<|UNK|>'] for token in result]
        return encoded_list

    def decode(self, token_ids):
        text = " ".join([self.inverse_vocab[i] for i in token_ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)
        return text
    