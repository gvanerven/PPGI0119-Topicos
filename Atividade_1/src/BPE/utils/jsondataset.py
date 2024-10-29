import os
import json
from collections.abc import Sequence, Iterable

class JSONDataset(Sequence, Iterable):
    def __init__(self, datadir):
        self.datadir = None
        self.lst_files = []
        self.pos = 0
        if os.path.isdir(datadir):
            self.datadir = datadir
            for file in os.listdir(self.datadir):
                if ".json" in file:
                    self.lst_files.append(os.path.join(datadir, file))
        else:
            print("Datadir is not a directory")
            return None
    
    def __iter__(self):
        #self.__init__(self.datadir)
        if self.pos >= len(self):
            self.pos = 0
        return self
        
    def __next__(self):
        if self.pos < len(self):
            item = self.__getitem__(self.pos)
            self.pos += 1
            return item
        else:
            raise StopIteration
        
    def __len__(self):
        return len(self.lst_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            if idx < 0: 
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise(IndexError, "The index (%d) is out of range." % idx)
            return self.__getfileinposition__(idx) 
        else:
            raise(TypeError, "Invalid argument type.")
        
    def __getfileinposition__(self, idx):
        if idx < len(self):
            with open(self.lst_files[idx], 'r') as f:
                content = json.load(f)
                return content['text']
        else:
            raise(IndexError, "The index (%d) is out of range in __getfileinposition__." % idx)
            
        
    