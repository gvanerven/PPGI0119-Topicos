from BPE_PPGI.utils.jsondataset import JSONDataset
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import nltk
import numpy as np
import random

class BigramBase():
    def __init__(self, tokenizer):
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        self.sp_matrix_bigram = np.ones((self.vocab_size, self.vocab_size))
        self.sp_count_tokens = np.full((self.vocab_size, 1), self.vocab_size)
        self.sp_matrix_bigram_prob = None

    def train(self, dataset):
        for text in tqdm(dataset):
            phrases = nltk.tokenize.sent_tokenize(text, language="portuguese")
            for phrase in phrases:
                tokens = self.tokenizer.encode(phrase)
                self.sp_count_tokens[tokens[0], 0] += 1
                for i in range(1, len(tokens)):
                    self.sp_matrix_bigram[tokens[i-1], tokens[i]] += 1
                    self.sp_count_tokens[tokens[i], 0] += 1
        self.sp_matrix_bigram_prob = self.sp_matrix_bigram / self.sp_count_tokens
        return self

    def generate_next(self, initial_text, maxn_ahead , stop_token_num, remove_last=True):
        tokens = list(self.tokenizer.encode(initial_text))
        if remove_last:
            tokens = tokens[:-1]
        count = 0
        next_token = None
        while count < maxn_ahead and next_token != stop_token_num:
            next_token = int(random.choices(range(0, self.vocab_size), self.sp_matrix_bigram_prob[:, tokens[-1]], k=1)[0])
            tokens.append(next_token)
            count += 1
        
        return self.tokenizer.decode(tokens)
        
if __name__ == "__main__":
    tokenizer_base_dir = os.path.join(os.path.abspath('.'), '..', '..', 'tokenizers')
    train_dir = os.path.join(os.path.abspath('.'), '..', '..', 'dados', 'train')
    train_ds = JSONDataset(train_dir)
    
    tokenizer_500 = AutoTokenizer.from_pretrained(os.path.join(tokenizer_base_dir, 'tokenizer_roberta_base_20_000'))
    bg = BigramBase(tokenizer=tokenizer_500)
    bg.train(train_ds)
    print("Sum mbg x2: ", 2*bg.sp_matrix_bigram.sum(), "Sum count: ", bg.sp_count_tokens.sum())
    print("Sum prob token 1: ", bg.sp_matrix_bigram_prob[:, 1].sum())
    print("Sum prob token 2: ", bg.sp_matrix_bigram_prob[:, 2].sum())
    print("Argmax token 2: ", bg.sp_matrix_bigram_prob[:, 2].argmax())
    print("Max token 2: ", bg.sp_matrix_bigram_prob[:, 2].max())
    print("Next Gen (Este é o início do texto sobre história de) + 50: ", bg.generate_next("Este é o início do texto sobre história", 50, 2))