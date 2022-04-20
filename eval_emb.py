from pytorch_transformers import GPT2Tokenizer
import torch
import numpy as np
import os
from fairseq.models.transformer_lm import TransformerLanguageModel
from embedding_eval.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_RG65, fetch_RW
from six import iteritems
import scipy


def get_tokvec(s, dict, emb, default=None):
    try:
        return emb[dict[s]]
    except KeyError as e:
        return default


def get_wordvec(w, unk_vec, dict, emb, tokenizer):
    vecs = []
    tokens = tokenizer.tokenize(w)
    for tok in tokens:
        vecs.append(get_tokvec(tok, dict, emb, default=unk_vec).numpy())
    vecs = np.array(vecs)

    return vecs.mean(axis=0)


def evaluate_similarity(tokenizer, token_dict, token_emb, X, y):
    """
        Calculate Spearman correlation between cosine similarity of the model
        and human rated similarity of word pairs

        Parameters
        ----------
        token_dict: dictionary of tokens

        token_emb: embedding of tokens

        X: array, shape: (n_samples, 2)
          Word pairs

        y: vector, shape: (n_samples,)
          Human ratings

        Returns
        -------
        cor: float
          Spearman correlation
        """

    missing_tokens = 0
    tokens = token_dict.keys()
    for query in X:
        for query_word in query:
            query_tokens = tokenizer.tokenize(query_word)
            for query_token in query_tokens:
                if query_token not in tokens:
                    missing_tokens += 1
    if missing_tokens > 0:
        print("Missing {} tokens. Will replace them with <unk> vector".format(missing_tokens))

    unk_vector = token_emb[token_dict['<unk>']]
    A = np.vstack(get_wordvec(word, unk_vector, token_dict, token_emb, tokenizer) for word in X[:, 0])
    B = np.vstack(get_wordvec(word, unk_vector, token_dict, token_emb, tokenizer) for word in X[:, 1])
    scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation


ckpt_path = './checkpoints/transformer_wikitext-103_bpe_agps2_0.03_gpu1'
ckpt_path = './checkpoints/transformer_wikitext-103_bpe1024_2'
ckpt_path = './checkpoints/transformer_wikitext-103_bpe_agps2_ul'

LangMo = TransformerLanguageModel.from_pretrained(ckpt_path, 'checkpoint6.pt')

token_dict = LangMo.tgt_dict.indices
token_emb = LangMo.models[0].decoder.output_projection.weight.detach()

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')

tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "RG65": fetch_RG65(),
    "RW": fetch_RW()
}
# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0],
                                                                                    data.X[0][1], data.y[0]))

# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name,
                                                           evaluate_similarity(tokenizer_gpt2, token_dict, token_emb,
                                                                               data.X, data.y)))
