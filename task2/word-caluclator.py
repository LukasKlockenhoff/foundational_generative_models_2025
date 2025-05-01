import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List, Tuple


class TransformerWordCalculator:
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

        vocab = list(self.model.tokenizer.get_vocab().keys())
        vocab = [w for w in vocab if not w.startswith('##')] # remove subword tokens
        self.vocab = vocab
        self.vocab_embeddings = self.model.encode(
            self.vocab,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def _parse_equation(self, eq: str) -> List[str]:
        eq = eq.replace(' ', '')
        eq
        tokens = re.findall(r'[+-]|[^\s+-]+(?:\s+[^\s+-]+)*', eq) # allow multi-word tokens
        return tokens

    def calculate(self, equation: str, top_k: int = 1) -> List[Tuple[str, float]]:
        tokens = self._parse_equation(equation)
        result_vec = None
        op = '+'

        for tok in tokens:
            if tok in ['+', '-']:
                op = tok
            else:
                embeddings = self.model.encode(tok, convert_to_numpy=True, normalize_embeddings=True)
                if result_vec is None:
                    result_vec = embeddings
                else:
                    if op == '+':
                        result_vec = result_vec + embeddings
                    else:
                        result_vec = result_vec - embeddings

        result_norm = result_vec / np.linalg.norm(result_vec)
        scores = np.dot(self.vocab_embeddings, result_norm)
        inputs = set([t.lower() for t in tokens if t not in ['+', '-']])
        idx_sorted = np.argsort(-scores)

        results = []
        for idx in idx_sorted:
            word = self.vocab[idx]
            if word.lower() in inputs:
                continue
            results.append((word, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

class GloveCalculator:
    def __init__(self,
                 glove_path: str,
                 embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.word_to_index = {}
        self.vocab = []
        self.embeddings = []

        with open(glove_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    continue
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                self.word_to_index[word] = idx
                self.vocab.append(word)
                self.embeddings.append(vec)

        self.embeddings = np.stack(self.embeddings)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.where(norms==0, 1, norms)

    def _parse_equation(self, eq: str) -> List[str]:
        tokens = re.findall(r'[+\-]|[\w\-]+', eq)
        return tokens

    def calculate(self,
                  equation: str,
                  top_k: int = 1) -> List[Tuple[str, float]]:
        tokens = self._parse_equation(equation)
        result_vector = None
        operator = '+'

        for token in tokens:
            if token in ['+', '-']:
                operator = token
            else:
                word = token.lower()
                if word not in self.word_to_index:
                    raise ValueError(f"Word '{word}' not in GloVe vocabulary.")
                word_embedding = self.embeddings[self.word_to_index[word]]
                if result_vector is None:
                    result_vector = word_embedding.copy()
                else:
                    if operator == '+':
                        result_vector += word_embedding
                    else:
                        result_vector -= word_embedding

        result_norm = result_vector / np.linalg.norm(result_vector)

        scores = np.dot(self.embeddings, result_norm)
        inputs = set([t.lower() for t in tokens if t not in ['+', '-']])
        idx_sorted = np.argsort(-scores)

        results = []
        for idx in idx_sorted:
            word = self.vocab[idx]
            if word in inputs:
                continue
            results.append((word, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1 and args[1] not in ['-g', '-t']:
        print("Word Vector Calculator")
        print("Usage: python word_calculator.py [-g <glove_path> <embeddings_dim>] | [-t <transformer_model>]")
        print("Default usage: python word_calculator.py")
        print("Default usage: uses transformer model 'all-MiniLM-L6-v2'")
        sys.exit(1)
    if len(args) > 1 and args[1] == '-g':
        if len(args) < 4:
            print("Usage: python word_calculator.py -g <glove_path> <embeddings_dim>")
            sys.exit(1)
        glove_path = args[2]
        embedding_dim = int(args[3])
        calc = GloveCalculator(glove_path, embedding_dim)
    elif len(args) > 1 and args[1] == '-t':
        if len(args) < 3:
            print("Usage: python word_calculator.py -t <transformer_model>")
            sys.exit(1)
        transformer_model = args[2]
        calc = TransformerWordCalculator(transformer_model)
    else:
        calc = TransformerWordCalculator()
    command = input(">>> ")
    while command != "exit" and command != "quit" and command != "q":
        try:
            print(calc.calculate(command, top_k=5))
        except Exception as e:
            print(f"Error: {e}")
        command = input(">>> ")
