import pickle

class BasicTokenizer():
    vocab = {}
    merges = {}
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids) -1: 
            if -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8")
        if verbose:
            print(tokens)
        tokens = list(map(int, tokens))
        # if verbose:
            # print(tokens)
        stats = self.get_stats(tokens)
        num_merges = vocab_size - 256
        ids = list(tokens)

        self.merges = {}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]        
        
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text    

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('../Tokenizer/taylorswift.txt', 'r', encoding='utf-8') as f:
    swift = f.read()

training_text = text[:int(3.0*len(swift))]
tokenizer = BasicTokenizer()
tokenizer.train(training_text, 276, True)
with open('../tokenizer.pkl', 'wb') as f:
    pickle.dump((tokenizer.vocab, tokenizer.merges), f)
