from collections import defaultdict
from itertools import pairwise, islice
import os
import regex as re
from multiprocessing import Pool
from typing import BinaryIO
import base64
import json
from typing import Dict, List, Tuple
from collections.abc import Iterable, Iterator
from heapq import heappush, heappop
import numpy as np



def process_chunk(path: str, start: int, end: int, splitter_pattern: re.Pattern, special_tokens: list[str]) -> defaultdict[tuple, int]:
    local_word_counts = defaultdict(int)
    
    with open(path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors='replace')

        text_parts = [chunk_text]
        for token in special_tokens:
            # For each special token, split every existing part further
            new_parts = []
            for part in text_parts:
                new_parts.extend(part.split(token))
            text_parts = new_parts
        
        # Now text_parts contains only the text BETWEEN special tokens
        for sub_chunk in text_parts:
            if not sub_chunk:
                continue
            p_iter = splitter_pattern.finditer(sub_chunk)
            for match in p_iter:
                word_tuple = tuple(match.group().encode("utf-8"))
                local_word_counts[word_tuple] += 1
            
    return local_word_counts

class BPETrainer:

    def __init__(self):
        self.pair_to_words = defaultdict(list)
        self.pair_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.vocab = {i : bytes([i]) for i in range(256)}
        self.special_tokens = []
        self.merges = []
        self.splitter = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.splitter_pattern = re.compile(self.splitter)
        self.heap = []
        self._inv_cache: dict[bytes, tuple[int, ...]] = {}
        self._inv_table = bytes.maketrans(bytes(range(256)), bytes(range(255, -1, -1)))


    def _lexinvert(self, tok: bytes) -> tuple[int, ...]:
        inv = tok.translate(self._inv_table)
        return (-len(tok), *inv)

    def save_vocab(self, file_path: str | os.PathLike) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("#\n")
            for idx in sorted(self.vocab):
                tok_b64 = base64.b64encode(self.vocab[idx]).decode("ascii")
                f.write(f"{idx}\t{tok_b64}\n")

    def save_merges(self, file_path: str | os.PathLike) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("#\n")
            for b1, b2 in self.merges:
                t1 = base64.b64encode(b1).decode("ascii")
                t2 = base64.b64encode(b2).decode("ascii")
                f.write(f"{t1} {t2}\n")

    def _pretokenize_parallel(self, data_path: str):
        num_processes = os.cpu_count()
        splitter_pattern = self.splitter_pattern

        with open(data_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        jobs = [(data_path, start, end, splitter_pattern, self.special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

        with Pool(processes=num_processes) as pool:
            list_of_dicts = pool.starmap(process_chunk, jobs)

        word_counts = defaultdict(int)
        for local_dict in list_of_dicts:
            for word, count in local_dict.items():
                word_counts[word] += count

        return word_counts



    def _lexinvert(self, tok: bytes) -> tuple[int, ...]:
        cached = self._inv_cache.get(tok)
        if cached is None:
            inv = tok.translate(self._inv_table)
            cached = self._inv_cache[tok] = (*inv, 255 - len(tok))
        return cached

    def _heap_push_pair(self, cnt: int, p0: int, p1: int) -> None:
        tok1, tok2 = self.vocab[p0], self.vocab[p1]
        heappush(
            self.heap,
            (
                -cnt,                 
                self._lexinvert(tok1),
                self._lexinvert(tok2),
                p0,                   
                p1,
            ),
        )


    def _heap_best_pair(self) -> tuple[int, int] | None:
        while self.heap:
            neg_cnt   = self.heap[0][0]     # first field
            p0, p1    = self.heap[0][-2:]   # last two fields
            real_cnt  = self.pair_counts.get((p0, p1), 0)
            if real_cnt and -neg_cnt == real_cnt:   # fresh
                return p0, p1
            heappop(self.heap)                       # stale → drop
        return None


    def _add_or_inc(self, b0: int, b1: int, delta: int = 1):
        pair = (b0, b1)
        new_cnt = self.pair_counts.get(pair, 0) + delta
        self._heap_push_pair(new_cnt, b0, b1)


    def _initialize_stats(self, input_path):
        # pretokenize
        word_counts = self._pretokenize_parallel(input_path)
        self.word_counts = word_counts
        print(len(word_counts))

        for word, count in self.word_counts.items():
            for pair in pairwise(word):
                self.pair_counts[pair] += count
                self.pair_to_words[pair].append(word)

        for (p1, p2), count in self.pair_counts.items():
            self._add_or_inc(p1, p2, 0)


    def _update_stats(self, old_word, new_word, count):
        self.word_counts[new_word] += count
        self.word_counts[old_word] -= count
        if self.word_counts[old_word] == 0:
            del self.word_counts[old_word]


        for p1, p2 in pairwise(old_word):
            pair = (p1, p2)
            self._add_or_inc(p1, p2, -1*count)
            self.pair_counts[pair] -= count


        for p1, p2 in pairwise(new_word):
            pair = (p1, p2)
            self._add_or_inc(p1, p2, count)
            self.pair_counts[pair] += count
            self.pair_to_words[pair].append(new_word)


    def train(self, input_path: str, vocab_size: int = 259, special_tokens: list[str] = [], verbose: bool = False):

        self.special_tokens = special_tokens
        for token_str in special_tokens:
            if token_str.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_str.encode("utf-8")
        self._initialize_stats(input_path)



        while len(self.vocab) < vocab_size:
            if verbose:
                print(f"Vocab size: {len(self.vocab)}")
    
            if not self.pair_counts:
                break

            best_pair = self._heap_best_pair()


            if self.pair_counts[best_pair] == 0:
                break    

            p1, p2 = self.vocab[best_pair[0]], self.vocab[best_pair[1]]

            self.merges.append((p1, p2))
            new_idx = len(self.vocab)
            self.vocab[new_idx] = p1 + p2
            affected_words = self.pair_to_words[best_pair].copy()

            for word in affected_words:
                if word not in self.word_counts:
                    continue
    
                count = self.word_counts[word]

                j = 0
                new_word = []
                n = len(word)
                while j < n:
                    if j < n - 1 and (word[j], word[j+1]) == best_pair:
                        new_word.append(new_idx)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
    
                self._update_stats(word, tuple(new_word), count)


        return self.vocab, self.merges
    
    

class Tokenizer: 

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.splitter = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pretokenization_pattern = re.compile(self.splitter)
        self.inverted_merge = {self.merges[i] : i for i in range(len(self.merges))}
        self.inverted_vocab = {self.vocab[i] : i for i in range(len(self.vocab))}
        if special_tokens == None: 
            self.special_tokens = {}
            self.special_token_pattern = None
        else: 
            self.special_tokens = {token: self.inverted_vocab[token.encode('utf-8')] for token in special_tokens}
            ## to handle special tokens like ["<|endoftext|>", "<|endoftext|><|endoftext|>"]. 
            ## Regex is annoying with priorities!!
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True) 
            escaped = [re.escape(t) for t in sorted_special_tokens]
            self.special_token_pattern = re.compile(f"({ '|'.join(escaped) })")

    @staticmethod
    def _load_vocab(file_path: str | os.PathLike) -> Dict[int, bytes]:
        vocab: Dict[int, bytes] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue                   # skip header / blank lines
                idx_str, tok_b64 = line.rstrip("\n").split("\t")
                idx = int(idx_str)
                vocab[idx] = base64.b64decode(tok_b64)
        return vocab
    
    @staticmethod
    def _load_merges(file_path: str | os.PathLike) -> List[Tuple[bytes, bytes]]:
        merges: List[Tuple[bytes, bytes]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                t1_b64, t2_b64 = line.rstrip("\n").split(" ")
                merges.append(
                    (base64.b64decode(t1_b64), base64.b64decode(t2_b64))
                )
        return merges

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _encode_word(self, word_bytes: list[int]) -> list[int]: 
        if not word_bytes:
            return []
        ids = [self.inverted_vocab[bytes([byte])] for byte in word_bytes]
        while len(ids) > 1: 
            possible_merges = [(self.inverted_merge.get((self.vocab[ids[i]], self.vocab[ids[i+1]]), float('inf')), i)
            for i in range(len(ids) - 1)]
            best_rank, pair_idx = min(possible_merges)

            if best_rank == float('inf'):
                break 
                    
            id1 = ids[pair_idx]
            id2 = ids[pair_idx+1]
            merged_bytes = self.vocab[id1] + self.vocab[id2]
            new_id = self.inverted_vocab[merged_bytes]                
            ids = ids[:pair_idx] + [new_id] + ids[pair_idx+2:]

        return ids
        
    
    def encode(self, text: str) -> list[int]: 
       
        if self.special_token_pattern == None:
            parts = [text]
        else: 
            parts = self.special_token_pattern.split(text)
            
        out = []
        for part in parts: 
            if part in self.special_tokens: 
                out.append(self.inverted_vocab[part.encode("utf-8")])
            else: 
                p_iter = re.finditer(self.splitter, part)
                for word in p_iter: 
                    word_bytes = bytes(word.group().encode("utf-8"))
                    if word_bytes in self.inverted_vocab:
                        out.append(self.inverted_vocab[word_bytes])
                    else: 
                        out +=(self._encode_word(word_bytes))

        return out

    def _get_chunk_boundary(self, chunk):
        
        last_safe_endpoint = 0
        if self.special_token_pattern != None: 
            special_matches = self.special_token_pattern.finditer(chunk)
            last_safe_endpoint = max(last_safe_endpoint, max((m.end() for m in special_matches), default=0))
        pretokenized_matches = self.pretokenization_pattern.finditer(chunk)
        last_safe_endpoint = max(last_safe_endpoint, max((m.end() for m in pretokenized_matches), default=0))        
        return last_safe_endpoint
    

    # Since we're only taking in an iterable of strings, we have no idea a priori how long the strings are. 
    # this function returns a generator that yields strings of length at most max_str_len. 
    def _normalize_iterable(self,iterable: Iterable[str], max_str_len: int) -> Iterator[str]:
        for large_string in iterable:
            if len(large_string) <= max_str_len:
                yield large_string
            else:
                start = 0
                while start < len(large_string):
                    end = start + max_str_len
                    yield large_string[start:end]
                    start = end

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # We're going to retrieve chunks of size at most 256 from each string
        # We will also retrieve at most 256 strings. 
        # Since characters in UTF 8 take betwee 1-4 bytes, we can assume the worst. 
        # The following will load at most one megabyte into memory at a time.
        chunk_size_items = 256 
        max_str_length = 256 
        left_over = ""
        safe_iterable = self._normalize_iterable(iterable, max_str_length)
        while True:
            chunk = "".join(list(islice(safe_iterable, chunk_size_items)))
            if chunk == "":
                break
            chunk = left_over + chunk
            chunk_end = self._get_chunk_boundary(chunk)
            left_over = chunk[chunk_end:]
 
            yield from self.encode(chunk[:chunk_end])

        yield from self.encode(left_over)

    def decode(self, ids: list[int]) -> str:
        byte_chunks = [self.vocab.get(i, b'') for i in ids]
        full_byte_sequence = b"".join(byte_chunks) 
        return full_byte_sequence.decode('utf-8', 'replace')
    
    def encode_to_file(self,load_path: str, save_path: str, *, flush_after=1_000_000):
        sampler = DocumentSampler(load_path)
        buf: list[int] = []                        
        with open(save_path, "wb") as out:
            for chunk in sampler.retrieve_all():
                buf.extend(self.encode(chunk))          
                if len(buf) >= flush_after:
                    print(f"Flushing buffer of size {len(buf)}")
                    np.asarray(buf, dtype=np.uint16).tofile(out)
                    buf.clear()                    
            if buf:                                
                np.asarray(buf, dtype=np.uint16).tofile(out)
        


class DocumentSampler: 

    def __init__(self, file_path: str, delimiter: str = "<|endoftext|>"):
        self.file_path = file_path
        self.delimiter = delimiter
        self.read_chunk_size = 1024 * 1024 # 1MB

    def retrieve_all(self) -> Iterator[str]:
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                # Read a large chunk of text from the file
                chunk = f.read(self.read_chunk_size)
                
                # If f.read() returns an empty string, we've reached the end of the file.
                if not chunk:
                    break
                
                yield chunk

    def retrieve_first_slice(self, num_documents: str):
        docs_yielded = 0
        leftover = ""
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            while docs_yielded < num_documents:
                chunk = f.read(self.read_chunk_size)
                
                if not chunk:
                    if leftover:
                        yield leftover
                    break
                
                buffer = leftover + chunk
                
                parts = buffer.split(self.delimiter)
                
                leftover = parts[-1]
                complete_docs = parts[:-1]
                
                for doc in complete_docs:
                    if docs_yielded >= num_documents:
                        return 
                    if doc: 
                        yield doc
                        docs_yielded += 1

    


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

