import json
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.merges = []  # Stores merge rules as tuples
        self.vocab = {}   # Stores final subwords
    
    def train(self, text):
        """Trains the BPE tokenizer on input text."""
        words = text.split()
        corpus = [" ".join(word) + " </w>" for word in words]  # Append end-of-word symbol
        
        # Count initial token frequencies
        token_freqs = Counter(corpus)
        
        while len(self.merges) < self.vocab_size:
            # Count pair frequencies
            pair_freqs = Counter()
            for word, freq in token_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i + 1])] += freq
            
            if not pair_freqs:
                break  # Stop if no more pairs can be merged
            
            # Select the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)
            
            # Replace pair in all words
            new_token_freqs = {}
            for word, freq in token_freqs.items():
                new_word = " ".join(word.split()).replace(" ".join(best_pair), "".join(best_pair))
                new_token_freqs[new_word] = freq
            token_freqs = new_token_freqs
        
        # Store vocabulary based on merges
        self.vocab = {f"{a}{b}": i for i, (a, b) in enumerate(self.merges)}

    def tokenize(self, text):
        """Tokenizes input text using trained BPE merges."""
        words = text.split()
        tokenized_text = []
        
        for word in words:
            symbols = list(word) + ["</w>"]
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                merge_candidates = [pair for pair in pairs if pair in self.merges]
                if not merge_candidates:
                    break
                best_pair = min(merge_candidates, key=self.merges.index)
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_symbols.append(symbols[i] + symbols[i + 1])
                        i += 2  # Skip merged pair
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            tokenized_text.append(" ".join(symbols))
        
        return tokenized_text

    def save_vocab(self, filename):
        with open(filename, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)
    
    def load_vocab(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.merges = data["merges"]


eng = BPETokenizer(vocab_size=200)
de = BPETokenizer(vocab_size=200)
de_eng = BPETokenizer(vocab_size=400)
with open("../data/german.txt", "r") as f:
    german = f.read()
with open("../data/english.txt", "r") as f:
    english = f.read()
try:
    with open("german_bpe.json", "r") as f:
        de.load_vocab("german_bpe.json")
except FileNotFoundError:
    print("German BPE vocab not found, training new model.")
    de.train(german)
    de.save_vocab("german_bpe.json")
try:
    with open("english_bpe.json", "r") as f:
        eng.load_vocab("english_bpe.json")
except FileNotFoundError:
    print("English BPE vocab not found, training new model.")
    eng.train(english)
    eng.save_vocab("english_bpe.json")
try:
    with open("german_english_bpe.json", "r") as f:
        de_eng.load_vocab("german_english_bpe.json")
except FileNotFoundError:
    print("German-English BPE vocab not found, training new model.")
    de_eng.train(german + english)
    de_eng.save_vocab("german_english_bpe.json")

while True:
    print("Enter 'exit' to quit.")
    sentence = input("Enter a sentence to tokenize: ")
    print("German BPE Tokenization:", de.tokenize(sentence))
    print("English BPE Tokenization:", eng.tokenize(sentence))
    print("German-English BPE Tokenization:", de_eng.tokenize(sentence))
    if sentence.lower() == 'exit':
        break

