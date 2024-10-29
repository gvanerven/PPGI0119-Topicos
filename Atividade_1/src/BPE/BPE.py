import sys
from .utils.jsondataset import JSONDataset
from multiprocessing import Pool

class BPEBase():
    def __init__(self, max_vocab_size):
        self.vocabulary = {idx: bytes([idx]) for idx in range(256)}
        self.max_merges = max_vocab_size - len(self.vocabulary)
        if self.max_merges < 0:
            self.max_merges = 0
            self.max_vocab_size = len(self.vocabulary)
        else:
            self.max_vocab_size = max_vocab_size
        self.merges = {}
        
    def tokenize(self):
        pass
    
    def encode(self):
        pass
    
    def decode(self):
        pass
    
    def train(self, dataset):
        tokens = []
        stats = {}
        for text in dataset:
            ids = list(map(int, text.encode("utf-8")))
            self.get_pair_stats(ids, stats)
            tokens.append(ids)
            
        for i in range(self.max_merges):
            aux = []
            if len(stats) == 0:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            stats = {}
            for tks in tokens:
                new_ids = self.merge_pair(tks, pair, idx) # na lista de tokens, trocar o par pelo idx
                self.get_pair_stats(new_ids, stats)
                aux.append(new_ids)
            tokens = aux
            self.merges[pair] = idx
            self.vocabulary[idx] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
            
        return self
    
    def get_pair_stats(self, ids, counts=None):
        if counts == None:
            counts = {}
        for pair in zip(ids, ids[1:]): 
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_pair(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

if __name__ == "__main__":
    dir = sys.argv[1]
    dataset = JSONDataset(dir)
    bpe = BPEBase(max_vocab_size=289)
    bpe.train(dataset)
    print(bpe.merges)
    print(bpe.vocabulary)