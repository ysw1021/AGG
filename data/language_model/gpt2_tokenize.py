from pytorch_transformers import GPT2Tokenizer
import os

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')

new_lines = []
with open('wikitext-103/wiki.train.tokens', "r") as f:
    worker_id = 0
    size = os.fstat(f.fileno()).st_size
    chunk_size = size
    offset = worker_id * chunk_size
    end = offset + chunk_size
    f.seek(offset)
    line = f.readline()
    while line:
        tokenized = tokenizer_gpt2.tokenize(line)
        if tokenized[0] == '<|endoftext|>':
            tokenized = []
        tokenized.append('\n')
        new_line = " ".join(tokenized)
        new_lines.append(new_line)
        if f.tell() > end:
            break
        line = f.readline()

print()
print('Writing preprocessed train data file...')
with open('wikitext-103/wiki.train.bpetokens', "w") as f:
    for l in new_lines:
        f.write(l)

new_lines = []
with open('wikitext-103/wiki.valid.tokens', "r") as f:
    worker_id = 0
    size = os.fstat(f.fileno()).st_size
    chunk_size = size
    offset = worker_id * chunk_size
    end = offset + chunk_size
    f.seek(offset)
    line = f.readline()
    while line:
        tokenized = tokenizer_gpt2.tokenize(line)
        if tokenized[0] == '<|endoftext|>':
            tokenized = []
        tokenized.append('\n')
        new_line = " ".join(tokenized)
        new_lines.append(new_line)
        if f.tell() > end:
            break
        line = f.readline()

print()
print('Writing preprocessed valid data file...')
with open('wikitext-103/wiki.valid.bpetokens', "w") as f:
    for l in new_lines:
        f.write(l)

new_lines = []
with open('wikitext-103/wiki.test.tokens', "r") as f:
    worker_id = 0
    size = os.fstat(f.fileno()).st_size
    chunk_size = size
    offset = worker_id * chunk_size
    end = offset + chunk_size
    f.seek(offset)
    line = f.readline()
    while line:
        tokenized = tokenizer_gpt2.tokenize(line)
        if tokenized[0] == '<|endoftext|>':
            tokenized = []
        tokenized.append('\n')
        new_line = " ".join(tokenized)
        new_lines.append(new_line)
        if f.tell() > end:
            break
        line = f.readline()

print()
print('Writing preprocessed test data file...')
with open('wikitext-103/wiki.test.bpetokens', "w") as f:
    for l in new_lines:
        f.write(l)
