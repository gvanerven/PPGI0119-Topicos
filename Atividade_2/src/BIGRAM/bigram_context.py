from BPE_PPGI.utils.jsondataset import JSONDataset
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import nltk

class Token():
    def __init__(self, token):
        self.token = token
        self.contexts = set()
        self.tokens_count_after = {}
        self.tokens_count_before = {}
        self.count_occurrence = 0
        self.pos_context = {}
        
class BigramContext():
    def __init__(self, tokenizer):
        self.tokens = {}
        self.tokenizer = tokenizer
        self.count_contexts = -1
        
    def train(self, dataset):
        for text in dataset:
            self.count_contexts += 1
            pos = -1
            phrases = nltk.tokenize.sent_tokenize(text, language="portuguese")
            for phrase in phrases:
                tokens = self.tokenizer.encode(phrase)
                if tokens[0] not in self.tokens:
                    self.tokens[tokens[0]] = Token(tokens[0])
                    
                before = None
                after = None
                
                for i in range(0, len(tokens)):
                    token = tokens[i]
                    pos += 1
                    
                    if i + 1 < len(tokens):
                        self.tokens[tokens[i+1]] = Token(tokens[i+1])
                        after = tokens[i+1]
                    else:
                        after = None
                    
                        
                    self.tokens[token].contexts.add(self.count_contexts)
                    if self.count_contexts not in self.tokens[token].pos_context:
                        self.tokens[token].pos_context[self.count_contexts] = [pos]
                    else:
                        self.tokens[token].pos_context[self.count_contexts].append(pos)
                        
                    if before != None:
                        self.tokens[token].tokens_count_before[before] = self.tokens[token].tokens_count_before.get(before, 0) + 1
                    before = token
                    
                    if after != None:
                        self.tokens[token].tokens_count_after[after] = self.tokens[token].tokens_count_after.get(after, 0) + 1
                        
                        
        return self




if __name__ == "__main__":
    tokenizer_base_dir = os.path.join(os.path.abspath('.'), '..', '..', 'tokenizers')
    train_dir = os.path.join(os.path.abspath('.'), '..', '..', 'dados', 'train')
    train_ds = JSONDataset(train_dir)
    
    tokenizer_500 = AutoTokenizer.from_pretrained(os.path.join(tokenizer_base_dir, 'tokenizer_roberta_base_500'))
    bg = BigramContext(tokenizer=tokenizer_500)
    bg.train(tqdm(train_ds))
    print(bg.tokens[0])