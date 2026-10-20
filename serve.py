"""
tinyserve: a small vLLM-style inference engine in one file.

This is a simpler implementation of Paged KV Cache with prefix caching, 
continuous batching with chunked prefill, FlashAttention (or a pure torch reference), 
CUDA graphs for decoding and an OpenAI-compatible HTTP Server
"""
import argparse
import asyncio
import glob
import json
import os
import queue
import random
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
import traceback
import warnings
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers import AutoTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM
from datasets import load_dataset

from fastapi import Request as HTTPRequest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except:
    FLASH_ATTN_AVAILABLE = False


@dataclass
class Config:

    model: str
    dtype: torch.dtype = torch.bfloat16 # bfloat16 inference by default
    max_model_len: str = 4096 # max prompt + output we accept
    attention_backend: str = "flash_attn" # "flash_attn" (fast) | "torch" (reference)

    # toks per kv-cache page. 0 = pick for backend, flash-attn only supports
    # pages that are multiples of 256, vllm actually has its own fork for
    # smaller pages (more efficient if we use shorter prompts on average)
    # the torch reference code can use anything
    block_size: int = 0 
    max_num_seqs: int = 128 # max requests we can have in a batch

    # token budget per step, 
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    use_cuda_graphs: bool = True
    seed: int = 0


    def __post_init__(self):
        # if flash_attn not available dont use
        if self.attention_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            warnings.warn("flash attention not available, default to torch")
            self.attention_backend = "torch"

        # set page size as small as possible based on backend
        if self.block_size == 0:
            self.block_size = 256 if self.attention_backend == "flash_attn" else 16

        # our manually looped torch implementation cant be cuda graphed
        if self.attention_backend == "torch":
            self.use_cuda_graphs = False  # its Python loop can't be captured

@dataclass
class SamplingParams:
    temperature: float = 1.0 # 0 means greedy decode
    top_p: float = 1.0
    top_k: int = -1          # -1 means off
    max_tokens: int = 64     # max tokens to gen
    ignore_eos: bool = False # normally we stop gen on eos token

    ### for testing ###
    forced_token_ids: list[int] | None = None # teacher forcing to compare log probs
    return_logprobs: bool = False

class Request:
    """
    This is the metadata that is needed by the scheduler about a single request

    The main field we care about is num_computed_tokens, it is how
    many token_ids already have K/V in the cache, each step the scheduler
    picks how many new tokens to process for this request:

    for example, if we have a prompt of 1000 tokens but a budget of 512, 
    then we will first do 512 tokens, then the remaining 488 (this is
    chunked prefill), afterwards during generation it is one token per step
    """

    def __init__(self, 
                 request_id: str, 
                 prompt_ids: list[int], 
                 params: SamplingParams):

        self.request_id = request_id
        self.prompt_len = len(prompt_ids)

        # prompt + generated so far
        self.token_ids = list(prompt_ids) 
        self.params = params

        # how many of those tokens already have K/V written
        self.num_computed_tokens = 0

        # this requests page table (what blocks contain the KV cache
        # for this specific request?)
        self.block_ids: list[int] = []


        # One has per full block, for prefix reuse
        self.block_hashes: list[int] = [] 
        self.num_cached_tokens = 0 # prefix-cache hits at admission
        self.logprobs: list[torch.Tensor] = []
        self.finished = False
        self.finish_reason: str | None = None

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def output_ids(self):
        # everything after the prompt is what was generated
        return self.token_ids[self.prompt_len:]

class KVCacheManager:
    """
    This manages which phyisical  KV-cache blocks each request owns

    The block manager o nly manages block ids, it doesnt touch GPU memory
    the model runner later converts those ids to actual kv addresses. 

    Each request sees its kv cache as a sequence of logical blocks, while
    the blocks themselves can live anywhere in GPU memory. For example
    with block_size = 4 and a request containing:
    
    [A B C D E F G H I]

    the request needs three logical blocks::

        logical block 0 -> tokens [A B C D]
        logical block 1 -> tokens [E F G H]
        logical block 2 -> tokens [I]

    The block manager might assign physical blocks:
    
        req.block_ids = [17, 42, 81]

    so tokens A-D live in physical block 17, E-H in block 42, and I in block 81.
    The physical blocks do not need to be contiguous.

    For token i:

        logical_block = i // block_size
        offset        = i % block_size
        block_id      = req.block_ids[logical_block]

    if the cache is viewed as one flat array the corresponding slot is:
    
        slot = block_id * block_size + offset

    For example, token F has i = 5:

        logical_block = 5 // 4 = 1
        offset = 5 % 4 = 1
        block_id = req.block_ids[1] = 42
        slot = 42 * 4 + 1 = 169

    Prefix Caching: full blocks can be reused by later requests (i.e. things
    like the system prompt can be reused for each request). Each full block is
    identified by a hash of its own parent blocks hash and its own tokens:

        block_hash = hash(parent_hash, tokens)

    The parent hash matters beacuse the same tokens can have different KV values
    when they occur after different prefixes. For example:

    Request A: 
        [system prompt] -> block 17 -> [A B C D] -> block 42 -> [E F G H] 
    
    Request B: 
        [system prompt] -> block 17 -> [A B C D] -> block 42 -> [E F G H] -> [X Y]


    Request B can reuse block 17 and 42 instead of recomputing their KV values
    it only needs a new block for [X Y]. 

    Only complete blocks are entered into the prefix cache, a partial final block
    cannot be reused until it is full. When an active request finishes, its
    cached blocks are not immediately erased, they become available for reuse
    and are placed on a least recently used (lru) free list while their hash-to-block
    mapping stays searchable. If a later request asks for the same prefix, the block can
    be taken back without recomputing its KV values. 

    Eventually, if GPU is memory is needed for a new block, the least recently used
    cached block is evicted and its physical storage is reused. At that point
    its old prefix KV is lost and must be recomputed if requested again.  

    """

    def __init__(self, 
                 num_blocks: int, 
                 block_size: int, 
                 enable_caching: bool):

        self.block_size = block_size
        self.enable_caching = enable_caching

        # therefore block 2 cannot be evicted while other requests are still using it
        self.ref_cnt = [0] * num_blocks

        # if a physical block contains a cached prefix, what prefxi does it represent
        self.block_hash : list[int | None] = [None] * num_blocks

        # ordered dict is double linked list, allows for O(1) pop-front
        # and O(1) removal from the middle. 
        # first we start at index 1, as block 0 is the null block used for padding tokens
        # their value doesnt matter as we mask it during attention anyway

        # This is effectively our LRU queue, it is a list of our unused blocks 
        self.free : OrderedDict[int, None] = OrderedDict((i, None) for i in range(1, num_blocks))

        # given this prefix hash, where is its kv data (hash -> physical block) ?
        self.hash_to_block: dict[int, int] = {}

    @property
    def num_free(self):
        return len(self.free)

    def _update_hashes(self, req: Request):
        """
        We will grow req.block_hashes here so there is one hash per completed
        block of token_ids. Effectively:

        hash_i = hash( (hash_{i-1} or None, tokens[i*B : (i+1)*B]) )

        Parent has is in the tuple so a block's identity commits the 
        entire prefix, now just its own tokens
        """

        # how many items per block
        bs = self.block_size

        # assume our requested tokens are 
        # [A B C D E F G H I]
        # and our block size is 4
        
        # The request needs hashes for complete blocks:
        # [A B C D]
        # [E F G H]
        # but not the final
        # [I]
        # as that block isnt full yet

        # so if we have 9 tokens and blocks size of 4, we have 2 
        # complete blocks and 1 incomplete. So we can log the hash
        # for the two compelte ones
        while len(req.block_hashes) < req.num_tokens // bs:

            # initially i = 0 (we start empty) and len(req.block_hashes) = 0
            i = len(req.block_hashes)

            # if it wasnt empty (after first iteration) then the parent is the
            # last block available for this specific request (-1)
            # i = 0: parent = None
            # i = 1: parent = hash(block 0)
            # i = 2: parent = hash(block 1)
            # which gives us the chain

            # we hash the parent along with the tokens
            # at i = 0 req.token_ids[0:4] is [A B C D], 
            # at i = 1 req.token_ids[4:8] gives [E F G H]
            # and our hashes become
            # H0 = hash(None, [A B C D])
            # H1 = hash(H0, [E F G H])

            parent = req.block_hashes[-1] if req.block_hashes else None
            req.block_hashes.append(hash((parent, tuple(req.token_ids[i * bs:(i + 1) * bs]))))
    
    def find_cached_prefix(self, req: Request):
        """
        This provides us the block ids of the longest cached prefix, 
        telling us how much of this requests prefix have we already
        computed?

        This is done by:

            1. refresh the hashes using the _update_hashes method
            2. walk the hashes left to rigth, stopping at the first miss
            3. cap at (num_tokens - 1) // B so atleast one token is 
               always computed (we need its logits)
        """
        if not self.enable_caching:
            return []

        # generate the requests hashes
        # suppose the request is [A B C D E F G H I J]
        # H0 = hash(None, [A B C D])
        # H1 = hash(H0,   [E F G H])
        # and remember from an earlier request this would be the same thing!
        # hash of the same input MUST give the same hash integer value 
        # regardless of the request it came from
        self._update_hashes(req)

        # we have to always leave atleast one token to compute. Suppose
        # num_tokens = 8 and block_size = 4, naively, 8 // 4 = 2 blocks. 
        # but instead we will do (8 - 1) // 4 = 1. To generate the next
        # token you need the logits at the last prompt position, which come
        # from running that token through the whole model. It shidden states 
        # passes through every layer and then the LM head, the KV cache only
        # keeps the K,V vectors of teh past tokens to later tokens can attend 
        # to them, it doesnt keep hidden states or logits. So if a requests
        # entire prompt came from the cache, the scheduler later would have
        # 0 tokens to run, nothing to feed the model, no logits, no first 
        # output tokens, num_tokens - 1 caps the cache hit so atleast the last
        # token is always run
        max_blocks = (req.num_tokens - 1) // self.block_size

        # Now we look up for matching hashes
        hits = []

        # suppose H0 -> block 17, H1 -> block 42, H2 -> missing
        # then our hits are block 17 and 42
        for h in req.block_hashes[:max_blocks]:
            b = self.hash_to_block.get(h)

            # cache hits have to be contiguous, we cant have 2 hits, a miss
            # and then another hit, as the K,V values of the third hit
            # are different when attended to the different miss
            if b is None:
                break

            hits.append(b)

        return hits

    def allocate(self,
                 req: Request, 
                 num_new_tokens: int, 
                 cached: list[int] = ()):
        """
        We need to ensure the page table covers the total

            total = already_computed + |cached| * B + num_new_tokens
            need  = ceil(total / B) - |req.block_ids| - |cached|

        basically: how many NEW physical pages do I need to pull
        off of the free list so this requests page table covers every 
        token that will exist after this step?
        """

        # first we calculate the total tokens
        # pretend we have  8 tokens already, 4 tokens in a cached prefix
        # and we want 6 new tokens, that is a total of 18 tokens. We need
        # ceil(18 / 4) = 5 blocks to fit that
        total = (
            req.num_computed_tokens 
            + len(cached) * self.block_size
            + num_new_tokens
        )

        # compute ceiling integer division
        total_needed_blocks = -(-total // self.block_size)

        # subtract out the number of blocks the request already has
        # as well as the blocks we are about to reuse. need can be 
        # 0 as in a previous step we may have allocated a whole block
        # and this decode step is filling unused slots in it
        need = total_needed_blocks - len(req.block_ids) - len(cached)

        # cached hits with ref_cnt == 0 are still sitting on the free list
        # attaching them removes them from free, so they look like
        # they consume a free slot even though we arent actually grabbing
        # a blank page
        #   need: brand new pages we will add
        #   reclaim: which hashed pages will you take off free
        reclaim = sum(
            1 for b in cached if self.ref_cnt[b] == 0
        )

        # if i need new blocks plus cached blocks that must be reclaimed from
        # the free pool, do I have enough free capacity?
        if need + reclaim > self.num_free:
            return False # if false then nothing to do until space opens up

        # claim cached blocks
        for b in cached:
            # suppose cached = [17, 42]
            if self.ref_cnt[b] == 0:

                # if block 17 was sitting on the free list
                # free = [17, 23, 31, 44]
                # we now just have free = [23, 31, 44]
                # which means that this block is no longer cached or fre, 
                # an active request owns it
                del self.free[b]

            # update the reference counter, if it was 0 we now have our new
            # reference. If it wasnt 0, lets say 1 as somthing else was using
            # it, now it would be 2, as 2 requests are using this block
            self.ref_cnt[b] += 1

            # the request now points at the cached physical block
            req.block_ids.append(b)

        # now we allocate new blocks we need
        for _ in range(max(need, 0)):
            b, _ = self.free.popitem(last=False) # take the first item

            # we are evicting b, but it still may be in our mapping
            # so we will delete it from there too (all the old metadata)
            # if we dont , then the recycled block would claim some information
            # is in there, when really that information has changed
            if self.block_hash[b] is not None:
                self.hash_to_block.pop(self.block_hash[b], None)
                self.block_hash[b] = None

            # give the block to the request
            # for example blocks 16 now has ref_count = 1 and 
            # is owned by the new request
            self.ref_cnt[b] = 1
            req.block_ids.append(b) 

        self._register_full_blocks(req, total)
        return True
    
    def _register_full_blocks(self, 
                              req: Request, 
                              num_tokens: int):
        """
        Did any of the blocks we just allocated become filled with known 
        tokens? if yes then we add them to the prefix cache
        """
        if not self.enable_caching:
            return 

        # make sure all relevant block hashes exist
        self._update_hashes(req)

        # Loop over only complete full blocks
        for i in range(min(num_tokens, req.num_tokens) // self.block_size):

            # get the block and hash
            b, h = req.block_ids[i], req.block_hashes[i]

            # register the hash to the block
            if self.block_hash[b] is None:
                self.block_hash[b] = h
                # we use set default as if the hash is already present
                # for some reason, there is no need to overwrite it
                self.hash_to_block.setdefault(h, b)

    def free_request(self, req: Request):
        # clear out everything at the end of a request
        # resetting the state back and returning the blocks
        # back for use by other requests
        for b in reversed(req.block_ids):
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free[b] = None
        req.block_ids = []

@dataclass
class ScheduledRequest:
    req: Request        # the live request
    num_computed: int   # KV present before this foward pass
    num_new: int        # how many tokens this forward will process

    @property
    def completes_prompt(self):
        """
        True: This step reaches the end of the current sequence (finish prefill
              or decode). The sampled next token is real and we append it
        
        False: chunked prefill stops mid prompt, you still run those tokens
               to write KV, but th elast positions sampled token is junk and we 
               throw it away, the next step continues the prompt
        
        num_tokens here is len(token_ids) which during prefill is just the prompt, 
        after a real sample, the scheduler's caller appends a token, so next step
        num_tokens grows by 1 and decode is num_new=1
        """
        return self.num_computed + self.num_new >= self.req.num_tokens

class Scheduler:
    """Called once per step: which requests run, and how many tokens each.

    1. Running requests go first (so decodes are never starved). If the KV
       cache is full, PREEMPT the newest running request: free its blocks and
       push it back to the waiting queue. It is recomputed later, and prefix
       caching usually makes that cheap.
    2. Then admit waiting requests while there is token budget, a free
       sequence slot, and KV space. Long prompts are CHUNKED to fit the budget.
    """

    def __init__(self, cfg: Config, kv: KVCacheManager):
        self.cfg = cfg
        self.kv = kv
        self.waiting: deque[Request] = deque() # FIFO of not yet admitted (or preempted) requests
        self.running: list[Request] = [] # currently holding KV pages (from oldest to newest)

    def add(self, req: Request):
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[ScheduledRequest]:

        # one token budget is shared by prefill chunks and decode tokens, so a long
        # prompt can never stall decodes for more than one steps worth of tokens
        budget = self.cfg.max_num_batched_tokens

        # plan for each request id how many tokens it runs this step
        plan: dict[str, int] = {}

        # if we evict anyone, dont admit new work
        preempted = False

        # 1. running requests
        i = 0
        # self.running is all of the currently running tasks, we loop through all of them
        # as long as we have some budget left
        while i < len(self.running) and budget > 0:

            # grab that request that is running
            req = self.running[i]

            # what this request still owes. 1 token in decode (the token just
            # sampled), or the rest of its prompt chunk. num_tokens counts 
            # prompt + everything generated so far
            n = min(req.num_tokens - req.num_computed_tokens, budget)

            # allocate() checks if a request can fit or not. 
            # running is in admit order (oldest at [0] newest at [-1])
            # we only ever evict from the newest end, so a request is never
            # preempted to make room for one younger than it. 
            while not self.kv.allocate(req, n):

                # pop will always evict the newest entry, this is so we can get the
                # older stuff done first before resuming the newer ones
                # for example, lets say we have requests [A B C], where C is the newest
                # and A is the oldest. i is currently on A, we try allocate(A, n), if
                # it fails, we will evict C and retry. If allocate(A, n) fails again, 
                # we will evict B and retry. If it still breaks (the victim is the req)
                # then we evict A and break, A is no longer running so the outer loop
                # stops too and A is not added to the plan
                victim = self.running.pop()

                # drops the victims pages. All shared prefix pages (ref_cnt > 1)
                # will survive, the rest return to the free queue (tail first). Hashes
                # are also kept so full blocks remain findable for future potential
                # cache hits
                self.kv.free_request(victim)

                # Its KV is gone, so it restarts from scratch next time. Its
                # token_ids (prompt + generated so far) are untouched, so the
                # recompute covers both. full blocks usually come back from the
                # prefix cache, the partial tail page is always recomputed.
                victim.num_computed_tokens = 0

                # we will put the victim at the front of the waiting list so it retries
                # before any brand new requests. We dont do this on this call, as preempted=True
                # will skip phase 2, we dont admit anyone new as our scheduler is currently
                # oversubscribed
                self.waiting.appendleft(victim)

                # victim always comes from index >= i and plan[] is only written after this 
                # loop, so this is always 0 in practice anyway, but we keep it just for 
                # completeness
                budget += plan.pop(victim.request_id, 0)

                # set flag that something was skipped
                preempted = True

                # req is the newest running request, there is nobody younger to evict 
                # anymore, so it preempts itself. If it is also the only running request
                # the cache simply cannot hold it and will retry forever. We break here
                # but a more complete version would verify that the cache fits the max_model_len
                # tokens, but we keep it simple here
                if victim is req:
                    break

            # the req preempted itself above, stop scheduling running requests, 
            # entirely this step.
            if req not in self.running:
                break

            # if req survived, we will reserve n tokens
            plan[req.request_id] = n

            # remove n tokens from budget
            budget -= n

            # move onto the next request
            i += 1

        # 2. waiting requests (skip if we just had to preempt as the cache is already tight)
        while (not preempted and self.waiting and budget > 0
               and len(self.running) < self.cfg.max_num_seqs):

            # whats currently on the waiting list
            req = self.waiting[0]

            # see if anythign is cached here to reuse (whole blocks only)
            cached = self.kv.find_cached_prefix(req)

            # Tokens not covered by cache hits, capped by the budget. Vecause
            # find_cached_prefix caps hits at (num_tokens - 1) // B, atleast one
            # token always runs, so the results can produce logits
            n = min(req.num_tokens - len(cached) * self.cfg.block_size, budget)

            # No preemption in this phase, an admitted request outranks a waiting one, 
            # then we break, not continue. The head of the queue blocks the rest (FIFO)
            # even if a shorter request behind it would ift
            if not self.kv.allocate(req, n, cached):
                break

            # if it fit, then we will remove the req from our list of waiting
            self.waiting.popleft()

            # KV that already exists for this request before the forward pass
            req.num_computed_tokens = len(cached) * self.cfg.block_size

            # this is for reporting only, keeps the best hit ever seen (a readmission
            # after preemption can be colder if pages were evicted meanwhile)
            req.num_cached_tokens = max(req.num_cached_tokens, req.num_computed_tokens)

            # newest = tail = first victim if phase 1 of a future step runs out of pages
            self.running.append(req)

            # reserve this requests slice of the current forward pass
            plan[req.request_id] = n
            budget -= n

        # 3. emit the batch in running order
        batch = []
        for req in self.running:
            # for every running request create a ScheduledRequest
            if req.request_id in plan:
                # Capture num_computed_tokens before advancing it: the runner
                # needs the old value for positions and slots, and
                # completes_prompt compares it against num_tokens.
                batch.append(ScheduledRequest(req, req.num_computed_tokens, plan[req.request_id]))

                # Advanced optimistically, before the model has run.
                req.num_computed_tokens += plan[req.request_id]

        return batch

    def finish(self, req: Request, reason: str) -> None:
        req.finished, req.finish_reason = True, reason
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self.kv.free_request(req)

@dataclass
class AttnMetadata:
    """
    Batch descriptor for one engine step, shared by every attention layer.

    q is packed [T, H, D] and has no batch dimension. These fields say how
    that stream splits into requests, where each new token's K/V is written
    and how to read each requests paged prefix. 

    There are two lengths per request:

        q_len: tokens that are run this step (rows of q)
        kv_len: keys that exist after the write (already computed + q_len)
                as we dont compute attention until the kv is fully updated 
                with new incoming items

    For example if we have 4 tokens in cache, and we run two more at positions
    4 and 5, our q_len = 2 (running 2 tokens) and the kv_len = 6

    T = sum of q_len over the batch
    B = num_seqs
    """

    # Flat cache address for each of the T new tokens.
    #   slot = block_id * block_size + (t % block_size)
    # Layer writes K/V to k_cache.view(-1, Hkv, D)[slot_mapping[t]]
    # before attention. Attention itself does not read this field.
    slot_mapping: torch.Tensor          # [T]

    # Prefix sums of this-step query lengths. Request i owns q[qsl[i]:qsl[i+1]].
    # qsl[0] = 0, qsl[B] = T. Flash-attn's cu_seqlens_q.
    #   A prefill 3 + B decode 1 + C prefill 4 ->  [0, 3, 4, 8]
    query_start_loc: torch.Tensor       # [B+1]

    # Prefix sums of full context lengths (kv_len), not q_len.
    # cu_k[i+1] - cu_k[i] == seq_lens[i]. Flash-attn's cu_seqlens_k.
    #   A kv=10, B kv=50, C kv=7  ->  [0, 10, 60, 67]
    # Does not mean K is packed densely, K stays paged. This only tells
    # the kernel how long each request's prefix is.
    cu_seqlens_k: torch.Tensor          # [B+1]

    # Page table. Row i is req.block_ids padded to max_blocks.
    #   block_table[i, t // bs] = physical page holding token t of request i
    # Only the first ceil(kv_len / bs) entries are live; the rest is
    # null-page padding and must not be gathered. Flash-attn reads this
    # in-kernel, torch_ref slices the live prefix and gathers in Python.
    block_table: torch.Tensor           # [B, max_blocks]

    # B. Live rows of block_table / entries in the cu_seqlens arrays.
    num_seqs: int

    # max(q_len_i) this step
    max_query_len: int

    # max(kv_len_i) this step
    max_seq_len: int

    # Host copies so torch_ref can index in Python without a GPU sync per request. 
    query_start_loc_cpu: list[int]      # [B+1]

    # Per-request kv_len on host. Not prefix-summed.
    # seq_lens_cpu[i] == cu_seqlens_k[i+1] - cu_seqlens_k[i]
    # torch_ref uses this directly as kv_len, flash-attn uses cu_seqlens_k.
    seq_lens_cpu: list[int] # [B]

def torch_ref_attention(
        q,         # [T ]
        k_cache, 
        v_cache, 
        meta, 
        scale
):
    """
    Paged attention.

    q:        [T, H, D]              packed queries for this step
    k/v_cache:[n_pages, bs, Hkv, D]  paged pool
    meta:     query_start_loc, seq_lens, block_table
    scale:    1/sqrt(D)

    For each request: gather its pages into a dense prefix, build a
    bottom-right causal mask, call SDPA. Slow (Python loop, copy of
    every context, host sync) but for understanding
    """
    out = torch.empty_like(q)
    
    # H = query heads, Hkv = KV heads (GQA if H > Hkv), bs = page size
    H, Hkv, bs = q.shape[1], k_cache.shape[2], k_cache.shape[1]

    # qsl[i]:qsl[i+1] = this request's slice of packed q
    # seq_lens[i]     = full context length after this step (kv_len)
    qsl, seq_lens = meta.query_start_loc_cpu, meta.seq_lens_cpu

    # for each sequence that has been packed
    for i in range(meta.num_seqs):
        s, e, kv_len = qsl[i], qsl[i + 1], seq_lens[i]

        # tokens processed for i this step
        q_len = e - s

        # live pages only: ceil(kv_len / bs). drop the padded tail of the row (null page 0)
        blocks = meta.block_table[i, : -(-kv_len // bs)].long()

        # Gather to a dense prefix
        #   k_cache[blocks] : [n_pages, bs, Hkv, D]
        #   flatten(0, 1)   : [n_pages*bs, Hkv, D]   pages stitched on the token axis
        #   [:kv_len]       : drop unused slots in the last page
        k = k_cache[blocks].flatten(0, 1)[:kv_len]
        v = v_cache[blocks].flatten(0, 1)[:kv_len]

        # SDPA wants H KV heads. GQA: each KV head is shared by H/Hkv query heads
        k = k.transpose(0, 1).repeat_interleave(H // Hkv, 0)
        v = v.transpose(0, 1).repeat_interleave(H // Hkv, 0)    

        # bottom right causal, this steps queries are teh last q_len positions
        # of a length kv_len sequence not 0 ... q_len - 1
        #   full prefill:  q_len == kv_len -> square causal mask
        #   chunked_prefill: q_len < kv_len -> suffix strip
        #   decode: q_len == 1 -> 1 row, all keys
        # if we have 4 kv positions already cached, and q_len = 2, then
        # our kv_len = 6 (as we will have 6 kv positions after this op) 
        # and our q_len = 2 for positions 4 and 5. Effectively, q at position
        # 4 can attend to all the kv before it (positions 0-3) and the new 
        # kv at the new position 4. Similarly, the q at position 5 can attend to
        # all the positions before it (0-3 + 4 just made) as well as the kv at 
        # the new position 5

        # keys    0  1  2  3  4  5
        # q@4     T  T  T  T  T  x  
        # q@5     T  T  T  T  T  T
        q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        mask = torch.arange(kv_len, device=q.device)[None, :] <= q_pos[:, None]

        # use SDPA
        o = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1), k, v, attn_mask=mask, scale=scale
        )
        out[s:e] = o.transpose(0, 1)

    return out

def flash_attention(q, k_cache, v_cache, meta, scale):
    """
    identical to the torch_ref but using flash_attn
    """
    return flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=meta.query_start_loc, cu_seqlens_k=meta.cu_seqlens_k,
        max_seqlen_q=meta.max_query_len, max_seqlen_k=meta.max_seq_len,
        softmax_scale=scale, causal=True, block_table=meta.block_table
    )

### SOME GLOBAL VARIABLES FOR SELECTING ATTENTION
ATTENTION_BACKENDS = {"flash_attn": flash_attention, "torch": torch_ref_attention}
ATTN_META: AttnMetadata | None = None
ATTN_FN = flash_attention

### REWRITE THE ATTENTION MECHANISM
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) \
                                 + self.eps) * self.weight.float()).to(x.dtype)
    
def rotary(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)

class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.H = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H

        ### qwen2 hardcodes true, but other models (llama, qwen3) expose attention_bias in config
        qkv_bias = getattr(cfg, "attention_bias", False) or cfg.model_type == "qwen2"
        self.q_proj = nn.Linear(cfg.hidden_size, self.H * self.D, bias=qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)
        self.o_proj = nn.Linear(self.H * self.D, cfg.hidden_size, bias=False)

        self.qk_norm = cfg.model_type == "qwen3"  # Qwen3 normalizes q and k per head
        if self.qk_norm:
            self.q_norm = RMSNorm(self.D, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.D, cfg.rms_norm_eps)

        self.layer_idx = layer_idx
        self.kv_cache: torch.Tensor | None = None  # [2, num_blocks, block_size, Hkv, D]

    def forward(self, x, cos, sin):
        T = x.shape[0] # x is passed in as packed

        q = self.q_proj(x).view(T, self.H, self.D)
        k = self.k_proj(x).view(T, self.Hkv, self.D)
        v = self.v_proj(x).view(T, self.Hkv, self.D)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        meta = ATTN_META # read from global

        if meta is None:  # profiling run: shapes only
            return self.o_proj(q.reshape(T, -1) * 0)

        # Write this step's new K/V into the page pool before attention.
        # k[:n] is packed [T, Hkv, D] in the same order as q.
        # slot_mapping[t] = page * block_size + offset for that token.
        # view(-1, Hkv, D) flattens (page, offset) onto dim 0 so
        #   cache.view(...)[slot] == cache[page, offset]
        # index_copy_(0, ...) does cache_flat[slot_mapping[t]] = k[t]
        # for every new token at once. Prefix pages not in slot_mapping
        # are left as-is.
        k_cache, v_cache = self.kv_cache[0], self.kv_cache[1]
        n = meta.slot_mapping.shape[0]
        k_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, k[:n])
        v_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, v[:n])

        # attend over each request's whole context through its block table
        out = ATTN_FN(q[:n], k_cache, v_cache, meta, self.D ** -0.5)

        return self.o_proj(out.reshape(n, -1))

### WRITE THE REST OF THE DECODER LAYER 
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.self_attn = Attention(cfg, i)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))

### WRITE THE MODEL
class Model(nn.Module):
    def __init__(self, cfg, max_position: int):
        super().__init__()

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.tie = getattr(cfg, "tie_word_embeddings", False)

        if not self.tie:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # RoPE table indexed by absolute position
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        assert rope.get("rope_type", "default") == "default", "only plain RoPE in this tutorial"
        
        theta = getattr(cfg, "rope_theta", None) or rope.get("rope_theta", 10000.0)
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

        inv_freq = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float64), inv_freq)
        self.register_buffer("cos", freqs.cos().float(), persistent=False)
        self.register_buffer("sin", freqs.sin().float(), persistent=False)

    def forward(self, input_ids, positions):
        cos, sin = self.cos[positions], self.sin[positions]
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def logits(self, hidden):
        w = self.embed_tokens.weight if self.tie else self.lm_head.weight
        return (hidden @ w.t()).float()

def load_model(name: str, dtype, max_position: int):
    """quick helper method to load weights into our model"""
    path = name if os.path.isdir(name) else snapshot_download(
        name, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    cfg = AutoConfig.from_pretrained(path)
    torch.set_default_dtype(dtype)
    with torch.device("cuda"):
        model = Model(cfg, max_position)
    torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, "pt", device="cuda") as fh:
                for key in fh.keys():
                    name_ = key.removeprefix("model.")
                    if name_ in params:
                        params[name_].copy_(fh.get_tensor(key))
    return model.eval()

### SAMPLER ###
def sample(logits, params: SamplingParams):
    """One batched call for mixed sampling settings.

    logits: [B, V]  next-token logits, one row per request that produced
                    a real sample this step (completes_prompt).
    params: length B, per-request temperature / top_k / top_p.
    """
    greedy = logits.argmax(-1)

    # if temp is 0 we assume greedy decode
    if all(p.temperature == 0 for p in params):
        return greedy.tolist()

    # what gpu
    dev = logits.device

    # Per-row temperature. temp==0 would divide by 0, so those rows use 1.0
    # here and get overwritten by greedy at the end via is_greedy.
    temp = torch.tensor([p.temperature or 1.0 for p in params], device=dev)
    logits = logits / temp[:, None] # [B, V]  higher temp -> flatter distribution

    # get vocab size
    V = logits.shape[-1]

    # top_k <= 0 means "off" -> keep the whole vocab for that row.
    top_k = torch.tensor([p.top_k if p.top_k > 0 else V for p in params], device=dev)
    top_p = torch.tensor([p.top_p for p in params], device=dev)

    # sort from largest to smallest logit (most to least likely), topk is a prefix
    # of the row, idx[b, r] is the original vocab_id of the rth loargest logit
    # so we can backtrack
    sorted_logits, idx = logits.sort(-1, descending=True)

    # ranks is 0 to V-1, where 0 cooresponds to the largest logit
    ranks = torch.arange(V, device=dev)[None, :]

    # get the probs themselves
    probs = sorted_logits.softmax(-1)

    # drop a sorted position if it fails topk or topp
    # ranks >= topk: keeps only the k largest (per row)
    # cumsum(p) - p > top_p, this token starts after the cumulative prob
    #                        has already reached top_p
    drop = (ranks >= top_k[:, None]) | (probs.cumsum(-1) - probs > top_p[:, None])
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

    # Scatter filtered logits back to vocab order, then softmax again. But now that 
    # dropped positions are -inf, they no longer contribute to the probs in softmax
    probs = torch.empty_like(logits).scatter_(-1, idx, sorted_logits).softmax(-1)

    # Gumbel-max: argmax(log p + Gumbel(0,1)) ~ Categorical(p).
    # G = -log Exp(1), so argmax(p / Exp(1)) is the same thing.
    gumbel = -torch.empty_like(probs).exponential_().log()
    sampled = (probs.log() + gumbel).argmax(-1)

    # if we wanted some samples to be greedy decode we replace that here
    is_greedy = torch.tensor([p.temperature == 0 for p in params], device=dev)
    
    return torch.where(is_greedy, greedy, sampled).tolist()

### MAIN RUNNER ###
class ModelRunner:
    """
    Turst the schedulers decisions into GPU tensors and runs the model

    For example, assume we have a block_size of 16A is decoding at position 37 with blocks
    [7, 3, 12], B is a new 5-token prompt with block [5]:
        input_ids       [a37, b0, b1, b2, b3, b4]   one flat batch, no padding
        positions       [37, 0, 1, 2, 3, 4]
        query_start_loc [0, 1, 6]                   request i = rows qsl[i]:qsl[i+1]
        seq_lens        [38, 5]                     how far each request attends
        block_table     [[7, 3, 12], [5, 0, 0]]
        slot_mapping    [197, 80, 81, 82, 83, 84]   12*16 + 37%16 = 197
        logits rows     [0, 5]                      last token of each request

    Additionally, we will leverage Cuda Graphs. The decode step of a smaller model is
    hundreds of tiny kernels, and launching them from Python costs more than running them, 
    we record the decode forward once per batch size into a CUDA graph and replay 
    it with a single call. Graphs bake in memory addresses, so all inputs live in a 
    persistent buffer that we overwrite each step, and batches are padded upto the 
    captured size (padding requests point at null block 0). Only decode batches are
    captured, their attention shape is fixed. Prefix (or chunked prefix) runs eagerly
    for simplicity 
    """

    def __init__(self, cfg):

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # load the model
        self.model = load_model(cfg.model, cfg.dtype, cfg.max_model_len)

        # get the model config
        mc = self.model.cfg

        # Get metadata
        self.H, self.Hkv = mc.num_attention_heads, mc.num_key_value_heads
        self.D = getattr(mc, "head_dim", None) or mc.hidden_size // self.H
        self.num_layers = mc.num_hidden_layers

        # set global var for attention
        global ATTN_FN
        ATTN_FN = ATTENTION_BACKENDS[cfg.attention_backend]

        # set max blocks per sequence (if our model supports lens upto 1024, and our
        # block size is 256, we can have at most 4 blocks)
        self.max_blocks_per_seq = -(-cfg.max_model_len // cfg.block_size)

        # Build persistent buffers for cuda graphs
        N, S = cfg.max_num_batched_tokens, cfg.max_num_seqs

        # helper method to create an input with a given shape
        z = lambda *shape, dt=torch.int32: torch.zeros(*shape, dtype=dt, device="cuda")

        # create some dummy inputs to act as our buffers
        self.input_ids, self.positions = z(N, dt=torch.int64), z(N, dt=torch.int64)
        self.slot_mapping = z(N, dt=torch.int64)
        self.query_start_loc, self.cu_seqlens_k = z(S + 1), z(S + 1)
        self.block_table = z(S, self.max_blocks_per_seq)

        # Place to store graphs
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    @torch.inference_mode()
    def allocate_kv_cache(self):
        """
        Run the largest possible batch without a cache, measure peak memory 
        create by the activations and intermediate computations and give
        the rest to KV (upto the set gpu_memory_utilization)
        """

        # Ensure our ATTN_META is just None for now
        global ATTN_META
        ATTN_META = None

        # reset stats
        torch.cuda.reset_peak_memory_stats()

        # Measure the high-water mark of weights + activations + workspace
        # for the largest packed step we will ever run.
        n = self.cfg.max_num_batched_tokens
        h = self.model(self.input_ids[:n], self.positions[:n])

        # Logits path sized for a full decode batch (one row per seq slot).
        self.model.logits(h[: self.cfg.max_num_seqs]).softmax(-1)

        # drop the tensor, peak already recorded
        del h

        # total = whole GPU
        # free = unused right now
        # reserved = PyTorch caching allocator arena
        # other = bytes held by someone else (or untracked)
        peak = torch.cuda.max_memory_allocated()
        free, total = torch.cuda.mem_get_info()
        other = total - free - torch.cuda.memory_reserved()

        # Leave (1 - gpu_memory_utilization) headroom, don't steal other,
        # and keep peak free so the next real forward still fits beside the cache.
        budget = total * self.cfg.gpu_memory_utilization - other - peak

        # Bytes for one page id across every layer:
        #   2 (K and V) * L * tokens/page * Hkv * D * 2 bytes (fp16/bf16)
        bytes_per_block = 2 * self.num_layers * self.cfg.block_size * self.Hkv * self.D * 2

        # Fit as many pages as the budget allows, but never more than
        # "every seq slot filled to max_model_len" plus page 0 (the null page).
        num_blocks = min(int(budget // bytes_per_block),
                         self.cfg.max_num_seqs * self.max_blocks_per_seq + 1)

        # After reserving page 0, the pool must still hold one max-length sequence.
        assert (num_blocks - 1) * self.cfg.block_size >= self.cfg.max_model_len, \
            "not enough GPU memory for even one max_model_len sequence"

        # One tensor per layer: [K|V, page, offset, kv_head, dim].
        # Same page-id space in every layer, so one block_table serves all of them.
        for layer in self.model.layers:
            layer.self_attn.kv_cache = torch.zeros(
                2, num_blocks, self.cfg.block_size, self.Hkv, self.D,
                dtype=self.cfg.dtype, device="cuda")

        print(f"[tinyserve] KV cache: {num_blocks} blocks x {self.cfg.block_size} tokens = "
              f"{num_blocks * self.cfg.block_size:,} tokens "
              f"({num_blocks * bytes_per_block / 2**30:.1f} GB)")
        
        return num_blocks

    def _prepare(self, batch: list[ScheduledRequest], pad_to: int = 0, for_graph: bool = False):
        """
        
        Pack this step into the persistent buffers the model / graphs read.

        Builds packed input_ids, positions, slot_mapping, query_start_loc,
        cu_seqlens_k, block_table, and AttnMetadata. Addresses of those
        buffers never change, which is what lets a CUDA graph replay.

        A decode batch is "how many live requests each produce one new
        token this step":

            A already has 40 tokens -> run token 41
            B already has 12 tokens -> run token 13
            C already has  7 tokens -> run token  8

            num_new = 1 for each
            T = 3
            q = [qA, qB, qC]          packed
            kv_len = 41, 13, 8        each attends over its own prefix

        Next step, if all three still run, T=3 again. A graph captured at
        batch size 4 serves this by adding one dummy row.

        pad_to > len(batch):
            Decode graphs are captured at a fixed batch size (T == B == pad_to,
            q_len == 1). A live decode batch smaller than that is padded with
            dummy 1-token requests: token 0, position 0, slot 0 (the null
            page). Their attention is ignored, logit_rows only lists real
            requests. Without this pad, T would change every step and the
            captured graph would be the wrong shape.

        Prefill / mixed batches (some num_new > 1) call this with pad_to=0
        and for_graph=False. T is the real token count and the forward
        runs eager. Graphs are decode-only because only then is T a
        function of batch size alone. One prefill chunk of 20 next to two
        decodes would make T=22 and miss the decode graph.
        """
        bs = self.cfg.block_size

        # packed streams we will copy onto the gpu buffers
        ids, pos, slots, logit_rows = [], [], [], []
        qsl, seq_lens = [0], [] # qsl starts at 0; we append the rest

        # page table for real + dummy rows
        bt = np.zeros((max(len(batch), pad_to), self.max_blocks_per_seq), dtype=np.int32)

        # prepare batches (each sample in the batch is one of the ScheduledRequest)
        for i, s in enumerate(batch):
            start, n = s.num_computed, s.num_new          # already-have-KV, this-step queries
            p = np.arange(start, start + n)               # absolute positions = q_pos
            blocks = np.asarray(s.req.block_ids)          # what are these requests block_ids

            # pack together each batch into a single stream
            ids.extend(s.req.token_ids[start:start + n])  # packed input ids
            # position embeds restart from 0 for every sample
            pos.extend(p.tolist())                        # RoPE / causal positions

            # slot = page * bs + offset, destination of index_copy_ for each new token
            slots.extend((blocks[p // bs] * bs + p % bs).tolist())

            qsl.append(qsl[-1] + n)                       # request i owns q[qsl[i]:qsl[i+1]]
            seq_lens.append(start + n)                    # kv_len after this step's write
            bt[i, : len(blocks)] = blocks                 # page-table row
            logit_rows.append(qsl[-1] - 1)                # last packed row -> sample logits here

        # Dummy decode reqeuests so T and B match the captured graph, 
        # slot 0 writes into the null page, outputs are never sampled
        # when doing graph capture, we dont pass in data, just padding
        # so all our q_len = 1 for all cases, which is the case of our
        # deocding anyway, so our graph capture will optimize the decoding 
        # cases for us
        for _ in range(len(batch), pad_to):
            ids.append(0), pos.append(0), slots.append(0)
            qsl.append(qsl[-1] + 1), seq_lens.append(1)

        T, B = len(ids), len(seq_lens)

        # persistent buffers: fixed addresses for CUDA graphs
        self.input_ids[:T].copy_(torch.tensor(ids))
        self.positions[:T].copy_(torch.tensor(pos))
        self.slot_mapping[:T].copy_(torch.tensor(slots))
        self.query_start_loc[: B + 1].copy_(torch.tensor(qsl))
        self.cu_seqlens_k[: B + 1].copy_(torch.tensor([0] + np.cumsum(seq_lens).tolist()))
        self.block_table[:B].copy_(torch.from_numpy(bt))

        # build the attn metadata
        max_seq = self.cfg.max_model_len if for_graph else max(seq_lens)

        meta = AttnMetadata(
            self.slot_mapping[:T],
            self.query_start_loc[: B + 1],
            self.cu_seqlens_k[: B + 1],
            self.block_table[:B],
            B,
            max(s.num_new for s in batch) if batch else 1,  # max_query_len
            max_seq,
            qsl, seq_lens,                                  # host copies for torch_ref
        )
        return T, meta, logit_rows

    @torch.inference_mode()
    def graph_sizes(self):
        # decode graph bucket sizes. Finegrained at bottom (1, 2, 4)
        # where small batches are common, then every 8 upto max_num_seqs, 
        # a live batch of N picks the smallest bucket >= N and pads the
        # rest with dummy 1 token requests on the null page
        sizes = [1, 2, 4] + list(range(8, self.cfg.max_num_seqs + 1, 8))
        return [s for s in sizes if s <= self.cfg.max_num_seqs]

    @torch.inference_mode()
    def capture_cuda_graphs(self):
        """
        A CUDA graph records kernel launches, and the exact buffer
        addresses they use, not the python call. Replay is just to fire 
        that list of kernels on those buffers again. 

        When running an op though, the very first call typically has more 
        overhead (like a cuBLAS workspace for this GEMM, scratch memory
        to store operation outputs, etc...) The second call just reuses those
        though, so we always run a single dummy call for that initial allocation
        followed up by capturing the second call. 
        """
        global ATTN_META
        t = time.time()

        # One workspace pool for every graph. Separate pools would each reserve
        # their own peak activation slab, sharing means we pay about max(peaks) 
        # rather than the sum of all our peaks
        pool = torch.cuda.graph_pool_handle()

        # Largest first, the first capture allocates the worst case scenario pool
        # and smaller graphs attach to that pool instead of growing it
        for bsz in reversed(self.graph_sizes()):

            # No real requests, pad_to=bsz dummy decode rows (q_len=1, slot 0).
            # T = B = bsz is now a compile-time constant for this graph.
            # for_graph=True bakes max_seq_len = max_model_len into metadata.
            T, meta, _ = self._prepare([], pad_to=bsz, for_graph=True)
            ATTN_META = meta

            # Warm-up on the same persistent buffers the graph will use.
            # Pays every lazy workspace alloc for this (T, B, max_seqlen).
            self.model(self.input_ids[:T], self.positions[:T])

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                # Recording: kernels only. input_ids / positions / ATTN_META
                # slices are captured by address. Next step we copy_ new
                # tokens into that same storage and replay.
                out = self.model(self.input_ids[:T], self.positions[:T])

            # output is the captured output buffer, replay overwrites it in place
            self.graphs[bsz] = (g, out)

        print(f"[tinyserve] captured {len(self.graphs)} decode CUDA graphs in {time.time() - t:.1f}s")

    @torch.inference_mode()
    def run(self, batch: list[ScheduledRequest]) -> tuple[list[int], torch.Tensor]:
        global ATTN_META

        # Graphs are decode-only. q_len is always 1 and T == bucket size.
        # A prefill chunk (num_new > 1) changes T every step. we don't
        # capture a graph per (T, B) pair, so those go eager.
        is_decode = all(s.num_new == 1 for s in batch)

        # Smallest bucket that fits this batch. None if N > every captured size
        # shouldn't happen if graph_sizes covers max_num_seqs.
        size = next((s for s in self.graph_sizes() if s >= len(batch)), None)

        # if we are decode only we can use cuda graph
        if self.graphs and is_decode and size is not None:
            # Pad to the captured shape. Real rows first, dummy decode rows
            # after. Dummies write slot 0 (null page) and are not in rows.
            T, ATTN_META, rows = self._prepare(batch, pad_to=size, for_graph=True)
            graph, out = self.graphs[size]
            graph.replay()                 # one launch, no allocs, same addresses
            hidden = out                   # the buffer captured above
        else:
            T, ATTN_META, rows = self._prepare(batch)   # true T, true max(seq_lens)
            hidden = self.model(self.input_ids[:T], self.positions[:T])

        # hidden is packed [T, ...]. rows indexes the last query of each
        # real request (the token we sample). Graph padding is skipped.
        logits = self.model.logits(hidden[rows])
        return sample(logits, [s.req.params for s in batch]), logits

### ENGINE
@dataclass
class Output:
    request_id: str
    text: str                   # full text so far
    delta: str                  # new text since the last output
    token_ids: list[int]
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_cached_tokens: int
    logprobs: list[torch.Tensor] | None = None   # [vocab] per generated token, on CPU

class Detokenizer:
    """
    Incrementally convert generated token IDs into text.

    Instead of decoding the entire output on every step (which would be
    O(n^2) over a long generation), we keep track of how much we've already
    decoded and only look at the new portion.

    We also avoid emitting a trailing U+FFFD ("�") because it can mean that
    a multi-byte UTF-8 character is incomplete and needs another token.
    """

    def __init__(self, tok):
        # The tokenizer used to convert token IDs -> text.
        self.tok = tok

        # Complete text that we've already returned to the caller.
        self.text = ""

        # Start of the token region we're currently decoding.
        self.prefix = 0

        # Number of token IDs we've already processed.
        self.read = 0

    def update(self, ids, final) -> str:
        # Decode the tokens we've already seen.
        #
        # Example:
        #   ids = [10, 20, 30, 40]
        #   prefix = 0
        #   read = 2
        #
        #   ids[prefix:read] = [10, 20]
        #
        # This gives us the text that was previously available.
        old = self.tok.decode(
            ids[self.prefix:self.read],
            skip_special_tokens=True,
        )

        # Decode everything from prefix up to the current end.
        # This lets us see what new text has appeared since the last update.
        new = self.tok.decode(
            ids[self.prefix:],
            skip_special_tokens=True,
        )

        # Don't emit anything if:
        # 1. new didn't get any longer, meaning no new text appeared.
        #
        # OR
        #
        # 2. new ends in "�" and this isn't the final update.
        #
        # "�" can indicate an incomplete multi-byte UTF-8 character.
        # We wait for another token before emitting it.
        if len(new) <= len(old) or (new.endswith("�") and not final):
            return ""

        # `old` is the text we already know about.
        # Everything after it is newly generated text.
        #
        # Example:
        #   old = "Hello"
        #   new = "Hello world"
        #   delta = " world"
        delta = new[len(old):]

        # Add the newly generated text to our complete output.
        self.text += delta

        # Move our bookkeeping forward:
        #
        # prefix becomes the old read position, and
        # read becomes the current number of tokens.
        #
        # This means the next call only needs to consider the new tokens.
        self.prefix, self.read = self.read, len(ids)

        # Return only the newly generated text.
        return delta

class Engine:
    """
    Put it all together!
    """
    def __init__(self, cfg: Config):

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.stop_ids = {self.tokenizer.eos_token_id}

        # some model could have multiple stop ids so just incase we make a list
        try:
            eos = GenerationConfig.from_pretrained(cfg.model).eos_token_id
            self.stop_ids |= set(eos if isinstance(eos, list) else [eos])
        except Exception:
            pass

        # If None is in here just remove it
        self.stop_ids.discard(None)

        # Set up the runner
        self.runner = ModelRunner(cfg)
        num_blocks = self.runner.allocate_kv_cache()
        if cfg.use_cuda_graphs:
            self.runner.capture_cuda_graphs()


        # Set up the kv cache and scheduler
        self.kv = KVCacheManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg, self.kv)

        # set up storage for the rest
        self.detok: dict[str, Detokenizer] = {}
        self.requests: dict[str, Request] = {}

    def add_request(self, 
                    prompt: str | list[int], 
                    params: SamplingParams,
                    request_id: str | None = None):

        # tokenize if we have to
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)

        # assert it fits
        assert 0 < len(ids) < self.cfg.max_model_len, "prompt too long"

        # unique id for request
        rid = request_id or uuid.uuid4().hex

        # save the request
        self.requests[rid] = req = Request(rid, ids, params)

        # Add to scheduler
        self.scheduler.add(req)

        # add to detokenizer
        self.detok[rid] = Detokenizer(self.tokenizer)

        return rid

    def abort(self, request_id: str) -> None:
        """Client went away: stop the request and give its KV blocks back."""
        req = self.requests.pop(request_id, None)
        if req is not None and not req.finished:
            self.scheduler.finish(req, "abort")
            self.detok.pop(request_id, None)

    def has_work(self):
        return self.scheduler.has_work() 

    def step(self):

        # Who runs this forward, and how many tokens each (phase 1+2 of schedule).
        batch = self.scheduler.schedule()

        # if nothing to do return nothing
        if not batch:
            return []

        # packed forward. tokens[i] is a sample from the last query row of batch[i]
        tokens, logits = self.runner.run(batch)

        outputs = []
        for i, (s, tok) in enumerate(zip(batch, tokens)):
            req = s.req

            # chunked prefill that stops mid-prompt, we wrote k/v for those tokens
            # but the last position is not the next output token, so we throw
            # these samples away
            if not s.completes_prompt:
                continue

            # Teacher forcing (equivalence tests). ignore the sample and
            # append a predetermined id so two engines walk the same path.
            # this is just for debugging
            forced = req.params.forced_token_ids
            if forced is not None:
                tok = forced[len(req.output_ids)] # output ids is tokens after prompt

            # get the full next token distribution  (one tensor per generated step)
            # this is also just for debugging
            if req.params.return_logprobs:
                req.logprobs.append(logits[i].log_softmax(-1).cpu())

            req.token_ids.append(tok) # grows output_ids (token_ids[prompt_len:])
            n_out = len(req.output_ids)

            # Stop conditions. finish() drops KV pages onto the free list
            # (hashes kept) and removes the request from running/waiting.
            if tok in self.stop_ids and not req.params.ignore_eos:
                self.scheduler.finish(req, "stop")
            elif n_out >= req.params.max_tokens or req.num_tokens >= self.cfg.max_model_len:
                self.scheduler.finish(req, "length")

            # incremental detokenization
            d = self.detok[req.request_id]
            delta = d.update(req.output_ids, req.finished)
            outputs.append(Output(
                req.request_id, d.text, delta, req.output_ids, req.finished,
                req.finish_reason, req.prompt_len, req.num_cached_tokens,
                req.logprobs if req.params.return_logprobs else None,
            ))
            if req.finished:
                del self.detok[req.request_id]
                del self.requests[req.request_id]

        return outputs

    def generate(self, prompts: list, params: SamplingParams | list[SamplingParams]):
        """Offline batch API, add everything, step until done."""
        params = params if isinstance(params, list) else [params] * len(prompts)
        rids = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        done: dict[str, Output] = {}
        while self.has_work():
            for o in self.step():
                if o.finished:
                    done[o.request_id] = o
        return [done[r] for r in rids]

### BUILD OPENAI COMPATIBLE SERVER
class AsyncEngine:
    """
    This class bridges the async web server (FastAPI) with a synchronous, compute heavy 
    model engine. It runs the model in a dedicated background thread, so that web requests
    are never blocked while the GPU is processing.
    """
    def __init__(self, engine: Engine):

        self.engine = engine

        # the inbox is a threadsafe queue. The web server drops requests here and the 
        # background engine picks them up.
        # # Messages look like: ("add", request_id, (prompt, params)) OR ("abort", request_id, None) 
        self.inbox = queue.Queue()

        # A dictionary mapping each active request_id to its specific web server event loop
        # and an asyncio.Queue. This is how the engine thread sends tokens back to the right user.
        self.streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}

        # Start the background thread immediately upon initialization.
        # daemon=True means this thread will automatically die if the main program exits.
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        """
        The main engine thread loop. It continuously checks for new web requests 
        and drives the model forward one step (token) at a time.
        """
        while True:

            # check if the engine is currently processing anything. If its
            # completely empty, we want the thread to sleep (block = True)
            # to save CPU. If its busy, we dont want it to sleep (block=False)
            block = not self.engine.has_work()

            # drain the inbox of all currently waiting messages
            while True:
                try:
                    # attempt to get a message from the webserver
                    kind, rid, payload = self.inbox.get(block=block)
                except queue.Empty:
                    # If inbox is empty, break out and move onto processing the model step
                    break

                # we recieved atleast one message, so the engine is definitely busy now
                # we set block=False so subsequent loop iterations dont get stuck waiting
                # if tehre are no more immediate messages
                block = False

                if kind == "add":
                    # A new request arrived. We add it to the engine's scheduler.
                    # It will be included in the next batch processed by engine.step().
                    self.engine.add_request(*payload, request_id=rid)
                else:
                    # The client disconnected ("abort"). Tell the engine to drop it from GPU memory.
                    self.engine.abort(rid)

            # at this point the scheduler has all new and existing requests
            try:
                # Tell the model to do one forward pass. It calculates the next token 
                # for every active request simultaneously (Continuous Batching).
                outputs = self.engine.step()
            except Exception:
                traceback.print_exc()
                continue

            # route the outputs back to the webserver
            for o in outputs:
                # Find where this output belongs
                stream = self.streams.get(o.request_id)

                if stream is not None:
                    loop, q = stream
                    # we are currently in the background thread, but q belongs to the webservers
                    # async event loop. asyncio is not threadsafe, so we use call_soon_threadsafe
                    # to safely push the output o into the webservers queue
                    loop.call_soon_threadsafe(q.put_nowait, o)

    async def generate(self, prompt, params: SamplingParams):
        """
        Called by FastAPI for each user request. It acts as an async generator, 
        yielding tokens as they arrive from the engine thread.
        """

        # Create a unique ID for this request
        rid = uuid.uuid4().hex      

        # create a personal queue to recieve tokens for this specific request
        q: asyncio.Queue = asyncio.Queue()

        # Register this request so the engine thread knows where to send the data
        self.streams[rid] = (asyncio.get_running_loop(), q)

        # drop the request into the engine threads inbox
        self.inbox.put(("add", rid, (prompt, params)))

        finished = False
        try:
            while not finished:

                # wait async for the engine thread to push a token into our queue
                o = await q.get()
                finished = o.finished
                yield o # stream the token to the user

        finally:
            # cleanup, this block runs when generation finishes
            self.streams.pop(rid, None)
            if not finished:  # the client disconnected mid-generation
                self.inbox.put(("abort", rid, None))

def build_app(engine: AsyncEngine, model_name: str):

    app = FastAPI()
    eng = engine.engine

    # Sampling defaults the model ships with in generation_config.json (Qwen3:
    # temperature 0.6, top_p 0.95, top_k 20), used when a request leaves them
    # out, like vLLM does. Only keys the file actually sets are used.
    defaults = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    try:
        shipped = GenerationConfig.from_pretrained(eng.cfg.model).to_diff_dict()
        defaults.update({k: shipped[k] for k in defaults if shipped.get(k) is not None})
    except Exception:
        pass

    def get(body, key, default):
        v = body.get(key)
        return default if v is None else v

    def params_from(body, prompt_len):
        # With no max_tokens the request may use the rest of the context
        # window, like the OpenAI API. A fixed default cuts long answers off
        # mid sentence (a thinking model can spend hundreds of tokens before
        # it even starts answering).
        room = eng.cfg.max_model_len - prompt_len
        max_tokens = get(body, "max_completion_tokens", get(body, "max_tokens", room))
        return SamplingParams(
            temperature=get(body, "temperature", defaults["temperature"]),
            top_p=get(body, "top_p", defaults["top_p"]),
            top_k=get(body, "top_k", defaults["top_k"]),
            max_tokens=min(max_tokens, room),
            ignore_eos=get(body, "ignore_eos", False),
        )

    def error(message: str, status: int = 400):
        return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}},
                            status_code=status)

    async def respond(prompt, body, chat):
        """
        Core response handler. Formats the raw engine outputs into the exact json structures
        expected by the OpenAI API clients
        """
        # Tokenize here rather than in the engine thread: we need the length
        # for max_tokens, and a prompt that does not fit must be rejected with
        # an error here. If it reached add_request in the engine thread, the
        # assert there would kill that thread and hang every request after it.
        ids = list(prompt) if isinstance(prompt, list) else eng.tokenizer.encode(prompt)
        if not 0 < len(ids) < eng.cfg.max_model_len:
            return error(f"prompt is {len(ids)} tokens, it must be between 1 and "
                         f"{eng.cfg.max_model_len - 1} (--max-model-len)")
        prompt = ids

        params, created = params_from(body, len(ids)), int(time.time())

        # prefix the ID to match OpenAIs convention
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex


        def choice(o: Output, text: str) -> dict:
            # fornat the actual generated text chunk, 
            # streaming chat endpoints use a delta key while others use messages
            if chat:
                key = "delta" if body.get("stream") else "message"
                return {"index": 0, key: {"role": "assistant", "content": text},
                        "finish_reason": o.finish_reason}
            return {"index": 0, "text": text, "finish_reason": o.finish_reason}

        def usage(o: Output) -> dict:
            # formats token counting metrics
            return {
                "prompt_tokens": o.num_prompt_tokens,
                "completion_tokens": len(o.token_ids),
                "total_tokens": o.num_prompt_tokens + len(o.token_ids),
                "prompt_tokens_details": {"cached_tokens": o.num_cached_tokens},
            }

        obj = "chat.completion" if chat else "text_completion"

        # Streaming path
        if body.get("stream"):

            async def events():
                # iterate over async generator from the engine
                async for o in engine.generate(prompt, params):
                    chunk = {
                        "id": rid,
                        "object": obj + (".chunk" if chat else ""),
                        "created": created,
                        "model": model_name,
                        "choices": [choice(o, o.delta)],
                    }
                    if o.finished: # we give a finsihed tag once done
                        chunk["usage"] = usage(o)
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        # Non-stream: drain until finished, return the accumulated text.
        # If the user didn't request a stream, silently drain the generator 
        # until finished, then return the entire accumulated result at once.
        final = None
        async for o in engine.generate(prompt, params):
            final = o
        return JSONResponse({
            "id": rid, "object": obj, "created": created, "model": model_name,
            "choices": [choice(final, final.text)], "usage": usage(final),
        })

    # Utility endpoint for load balancers to check if server is alive
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Allows clients to query which model is currently loaded
    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": 0,
                                            "owned_by": "tinyserve",
                                            "max_model_len": eng.cfg.max_model_len}]}

    # Custom endpoint to monitor internal engine health (queue sizes, VRAM usage)
    @app.get("/metrics")
    async def metrics():
        s = engine.engine.scheduler
        return {
            "running": len(s.running),
            "waiting": len(s.waiting),
            "free_kv_blocks": engine.engine.kv.num_free,
        }

    # Standard completions endpoint (Legacy style, no chat roles)
    @app.post("/v1/completions")
    async def completions(req: HTTPRequest):
        body = await req.json()
        return await respond(body["prompt"], body, chat=False)

    # Modern Chat Completions endpoint (Requires roles like system/user/assistant)
    @app.post("/v1/chat/completions")
    async def chat(req: HTTPRequest):
        body = await req.json()

        # OpenAI clients may send content as a list of parts
        # ([{"type": "text", "text": ...}, {"type": "image_url", ...}]). This is
        # a text-only server, so keep the text parts and join them.
        messages = [
            {**m, "content": "".join(p.get("text", "") for p in m["content"]
                                     if p.get("type") == "text")}
            if isinstance(m.get("content"), list) else m
            for m in body["messages"]
        ]

        # Uses the model's tokenizer to convert the JSON array of chat messages
        # into the specific raw string format the model was trained on
        # (e.g., Llama 3 or ChatML formats).
        prompt = eng.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **(body.get("chat_template_kwargs") or {}),
        )
        return await respond(prompt, body, chat=True)

    return app

def kv_cache_demo():
    B = 4
    mgr = KVCacheManager(num_blocks=8, block_size=B, enable_caching=True)

    def cached_pages():
        return [b for b, h in enumerate(mgr.block_hash) if h is not None]

    def dump(label, *reqs):
        print(f"\n=== {label} ===")
        print(f"  free (LRU front): {list(mgr.free)}")
        print(f"  ref_cnt:          {mgr.ref_cnt}")
        print(f"  hashed pages:     {cached_pages()}")
        for r in reqs:
            print(f"  {r.request_id}: {r.num_tokens} tok, computed={r.num_computed_tokens}, "
                  f"pages={r.block_ids}, hashes={len(r.block_hashes)}")

    dump("fresh (page 0 reserved)")

    # 1. chunked prefill
    a = Request("A", [10, 11, 12, 13, 14, 15], SamplingParams(max_tokens=4))
    print("\nA: 6-token prompt, budget 4. admit -> lookup once.")
    print("  hits", mgr.find_cached_prefix(a), mgr.allocate(a, 4))
    a.num_computed_tokens = 4
    dump("A chunk 1, page 1 full, hashed", a)

    print("\nA still live: rest of prompt. no lookup.")
    print("  allocate +2", mgr.allocate(a, 2))
    a.num_computed_tokens = 6
    dump("A prompt done, page 2 half full, not hashed", a)

    # 2. decode
    print("\nA decodes two tokens into the slots in page 2.")
    for t in (16, 17):
        a.token_ids.append(t)
        mgr.allocate(a, 1)
        a.num_computed_tokens += 1
    dump("A = 8 tokens, page 2 now hashed", a)

    # 3. B shares first block only
    b = Request("B", [10, 11, 12, 13, 77, 88], SamplingParams(max_tokens=4))
    hits = mgr.find_cached_prefix(b)
    print("\nB admit: same first 4 tokens.")
    print("  hits", hits, mgr.allocate(b, 2, cached=hits))
    b.num_computed_tokens = len(hits) * B + 2
    dump("B attached page 1, new page for 77,88", a, b)
    print(f"  page {hits[0]} ref_cnt={mgr.ref_cnt[hits[0]]} (A and B)")

    # 4. why max_blocks uses num_tokens-1
    c = Request("C", a.token_ids[:8], SamplingParams(max_tokens=4))
    print("\nC admit: exactly A's 8 tokens (2 full hashed blocks).")
    print("  hits", mgr.find_cached_prefix(c), ", one block, not two.")
    print("  (num_tokens-1)//B leaves one token to run; cache has K/V, not logits.")

    # 5. free A
    print("\nA finishes. tail page freed first.")
    mgr.free_request(a)
    dump("A gone, page 2 free but still hashed; page 1 held by B", b)

    # 6. eviction
    d = Request("D", list(range(200, 220)), SamplingParams(max_tokens=4))
    h2 = a.block_hashes[1]
    print("\nD needs 5 pages; free =", list(mgr.free))
    print("  A's 2nd block lives on page", mgr.hash_to_block.get(h2), "(front of free)")
    print("  allocate D +20", mgr.allocate(d, 20))
    d.num_computed_tokens = 20
    dump("D took every free page", b, d)
    print("  A's 2nd block now at", mgr.hash_to_block.get(h2), "(evicted with page 2)")
    print("  page 1 lives: B still holds it.")

    # 7. OOM on pages
    e = Request("E", [1, 2, 3, 4], SamplingParams(max_tokens=4))
    print("\nE: no free pages.")
    print("  allocate E +4", mgr.allocate(e, 4), ", no mutation")

def attention_demo():

    torch.manual_seed(0)

    dev, dt = "cuda", torch.bfloat16
    H, Hkv, D = 8, 2, 64 # 8 query heads sharing 2 KV heads (GQA)
    PAGE = 256 # upstream flash-attn: pages of 256 tokens
    scale = D ** -0.5

    # what each request contributes to this step
    spec = [("A  decode", 1, 600), ("B  fresh prompt", 5, 0), ("C  prefill chunk", 4, 300)]

    # set up metadata
    seq_lens = [q + c for _, q, c in spec]
    n_pages = [-(-s // PAGE) for s in seq_lens]
    num_blocks = 1 + sum(n_pages)         # page 0 is the null page, never used

    # randomly scatter the pages and build the block table
    perm = (torch.randperm(num_blocks - 1) + 1).tolist()
    kv = torch.zeros(2, num_blocks, PAGE, Hkv, D, device=dev, dtype=dt)
    k_cache, v_cache = kv[0], kv[1]

    block_table = torch.zeros(len(spec), max(n_pages), dtype=torch.int32)
    q_rows, k_new, v_new, slots, taken = [], [], [], [], 0
    for i, ((_, q_len, ctx), np_) in enumerate(zip(spec, n_pages)):

        # grab the pages for this specific page
        pages = perm[taken:taken + np_]

        # iterate number of pages used up
        taken += np_

        # assign the block table (padded to max blocks)
        block_table[i, :np_] = torch.tensor(pages, dtype=torch.int32)
        slot = lambda t: pages[t // PAGE] * PAGE + t % PAGE

        # Context tokens are already in the cache from earlier steps.
        # lets just fill it with random stuff as an example
        for t in range(ctx):
            k_cache.view(-1, Hkv, D)[slot(t)] = torch.randn(Hkv, D, device=dev, dtype=dt)
            v_cache.view(-1, Hkv, D)[slot(t)] = torch.randn(Hkv, D, device=dev, dtype=dt)

        # This step's tokens: the model would write these, then attend.
        q_rows.append(torch.randn(q_len, H, D, device=dev, dtype=dt))
        k_new.append(torch.randn(q_len, Hkv, D, device=dev, dtype=dt))
        v_new.append(torch.randn(q_len, Hkv, D, device=dev, dtype=dt))
        slots += [slot(t) for t in range(ctx, ctx + q_len)]

    # queries concatenated to [T, H, D]
    q = torch.cat(q_rows)

    # store our slot mapping (which global index does each token go to)
    slot_mapping = torch.tensor(slots, device=dev)

    # build the query starts ([0 1 6 10])
    qsl = [0]
    for _, q_len, _ in spec:
        qsl.append(qsl[-1] + q_len)

    # build our kv lens [0, 601, 606, 910]
    cu_k = [0]
    for s_ in seq_lens:
        cu_k.append(cu_k[-1] + s_)
    
    meta = AttnMetadata(
        slot_mapping=slot_mapping,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=dev),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=dev),
        block_table=block_table.to(dev),
        num_seqs=len(spec), max_query_len=max(q for _, q, _ in spec),
        max_seq_len=max(seq_lens), query_start_loc_cpu=qsl, seq_lens_cpu=seq_lens)

    print("One step, three requests, one flat batch of", q.shape[0], "tokens\n")
    print(f"  {'request':18} {'q_len':>6} {'ctx':>5} {'seq_len':>8}  pages (scattered)")
    for i, (label, q_len, ctx) in enumerate(spec):
        pages = block_table[i, :n_pages[i]].tolist()
        print(f"  {label:18} {q_len:>6} {ctx:>5} {seq_lens[i]:>8}  {pages}")
    print(f"\n  query_start_loc {qsl}   (request i = rows qsl[i]:qsl[i+1])")
    print(f"  cu_seqlens_k    {cu_k}   (prefix sum of seq_lens)")
    print(f"  slot_mapping    {slots[:6]}{' ...' if len(slots) > 6 else ''}")

    # The layer writes this step's K/V into the pages, then attends.
    k_cache.view(-1, Hkv, D).index_copy_(0, slot_mapping, torch.cat(k_new))
    v_cache.view(-1, Hkv, D).index_copy_(0, slot_mapping, torch.cat(v_new))

    out_ref = torch_ref_attention(q, k_cache, v_cache, meta, scale)
    out_fa = flash_attention(q, k_cache, v_cache, meta, scale)

    same = torch.allclose(out_ref.float(), out_fa.float(), atol=2e-2, rtol=2e-2)
    print("Matches:", same)

def build_shared_prefix(tok, alpaca, num_tokens: int, seed: int) -> str:
    """
    A long system prompt every request starts with and instruction plus few shot
    examples drawn from alpaca until we hit atleast num_tokens. This is obviously
    a little weird as it will probably mess with the final answers, but we dont
    really care about generation quality, simply throughput and this is an
    easy way to test prefix cache. 
    """
    rng = random.Random(seed + 1)
    text = ("You are a helpful assistant. Answer the user's request directly and "
            "concisely. Here are some examples of good answers:\n")
    while len(tok(text)["input_ids"]) < num_tokens:
        x = alpaca[rng.randrange(len(alpaca))]
        text += (f"\n### Request\n{x['instruction']} {x['input'] or ''}".rstrip()
                 + f"\n### Answer\n{x['output']}\n")
    return text

def load_benchmark_prompts(tok, num_prompts: int, edges: list[int], seed: int, chat: bool,
                           prefix_tokens: int = 0):
    """
    Real instructions from Alpaca + Dolly, sampled equally across length bins.

    With chat=True each instruction is wrapped in the model's chat template, so
    the model answers and stops at its end-of-turn token. Raw text has no
    natural end, the model just keeps writing until max_tokens.

    With prefix_tokens > 0 every prompt also starts with the same system
    prompt of about that many tokens. Bins are still decided by the request's
    own length, without the shared prefix.
    """

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    texts = ([x["instruction"] + (x["input"] or "") for x in alpaca]
             + [x["instruction"] + " " + (x["context"] or "") for x in dolly])

    def encode(texts, system=""):
        # tokenize with the model's own tokenizer, so lengths are in the units
        # the engines actually pay for (template tokens included)
        if chat and tok.chat_template:
            msgs = [([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": t}] for t in texts]
            texts = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
            return tok(texts, add_special_tokens=False)["input_ids"]
        return tok([system + "\n\n" + t if system else t for t in texts],
                   add_special_tokens=True)["input_ids"]

    # bin every request by its own length, then sample equally per bin
    lens = [len(p) for p in encode(texts)]
    names = ["short", "medium", "long"][:len(edges) - 1]
    by_bin = {name: [i for i, n in enumerate(lens) if lo <= n < hi]
              for name, lo, hi in zip(names, edges, edges[1:])}

    rng = random.Random(seed)
    per_bin = [num_prompts // len(names) + (i < num_prompts % len(names)) for i in range(len(names))]
    picked = []
    for name, k in zip(names, per_bin):
        assert len(by_bin[name]) >= k, f"bin {name} only has {len(by_bin[name])} prompts"
        picked += [(name, i) for i in rng.sample(by_bin[name], k)]
    rng.shuffle(picked)

    # only now prepend the shared system prompt (if any) to the chosen requests
    system = build_shared_prefix(tok, alpaca, prefix_tokens, seed) if prefix_tokens else ""
    prompts = encode([texts[i] for _, i in picked], system)
    return [b for b, _ in picked], prompts, names

@torch.inference_mode()
def hf_static_generate(model, tok, prompts: list[list[int]], max_tokens: int,
                       batch_size: int, stop_ids: set[int], ignore_eos: bool):
    """
    The standard method: cut the prompts into fixed batches in arrival order,
    left pad each batch to its longest prompt and call model.generate. A batch
    only finishes when its slowest row does, and the next batch cannot start
    until then, so a short request that lands next to a long one waits for it.

    Returns per-request (num_output_tokens, finish_time) plus waste counters:
        pad_prompt:  prompt slots that were padding
        pad_decode:  decode slots spent on rows that had already finished
    """
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n_out, done_at = [0] * len(prompts), [0.0] * len(prompts)
    pad_prompt = pad_decode = 0
    t0 = time.perf_counter()

    ### loop through the prompts in fixed sized batches
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]

        # get the longest sample in this batch
        L = max(len(p) for p in chunk)

        # left pad, generation continues on the right
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, L - len(p):] = torch.tensor(p)
            mask[j, L - len(p):] = 1

        # built in generate
        gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), do_sample=False,
                             max_new_tokens=max_tokens,
                             min_new_tokens=max_tokens if ignore_eos else 0,
                             eos_token_id=list(stop_ids), pad_token_id=pad_id)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        # count each row's real output, up to and including its first stop
        # token (the same way the engine counts it)
        out = gen[:, L:].tolist()
        steps = len(out[0]) # total generations
        for j, row in enumerate(out):
            k = len(row) if ignore_eos else next(
                (s + 1 for s, tok_id in enumerate(row) if tok_id in stop_ids), len(row))
            n_out[i + j], done_at[i + j] = k, t
            pad_decode += steps - k
        pad_prompt += L * len(chunk) - sum(len(p) for p in chunk)

    return n_out, done_at, pad_prompt, pad_decode

def tinyserve_generate(engine: Engine, prompts: list[list[int]], max_tokens: int, ignore_eos: bool):
    """
    Same prompts, all handed to the engine up front. This is Engine.generate
    but it also records when each request finished: continuous batching lets
    a short request leave the batch (and free its pages) as soon as it is
    done, and a waiting request takes its place on the very next step.

    Also returns how many prompt tokens of each request came from the prefix
    cache instead of being computed.
    """
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=ignore_eos)
    rids = [engine.add_request(p, params) for p in prompts]
    done: dict[str, tuple[int, float, int]] = {}
    t0 = time.perf_counter()
    while engine.has_work():
        for o in engine.step():
            if o.finished:
                done[o.request_id] = (len(o.token_ids), time.perf_counter() - t0,
                                      o.num_cached_tokens)
    return ([done[r][0] for r in rids], [done[r][1] for r in rids],
            [done[r][2] for r in rids])

def benchmark(a):
    """
    Real prompts (equal per length bin), chat formatted, all starting with
    the same long system prompt, all available at t=0, greedy, stop at EOS or
    max_tokens. Every run uses the same weights, dtype, GPU, prompts and stop
    tokens. Three runs:

        HF static:                   fixed batches, padded, each batch waits
                                     for its slowest row
        tinyserve (no prefix cache): continuous batching over a paged KV
                                     cache, chunked prefill, CUDA graphs for
                                     decode; every prompt is fully computed
        tinyserve (prefix cache):    the same, but the shared system prompt is
                                     computed once and its KV pages are reused

    Each run is freed before the next one starts, so each gets the whole card.
    """
    tok = AutoTokenizer.from_pretrained(a.model)
    stop_ids = {tok.eos_token_id}

    # grab eos token
    eos = GenerationConfig.from_pretrained(a.model).eos_token_id
    stop_ids |= set(eos if isinstance(eos, list) else [eos])

    # dont need none its sometimes there
    stop_ids.discard(None)

    # get benchmark prompts
    edges = [int(x) for x in a.bench_bins.split(",")]
    bins, prompts, names = load_benchmark_prompts(tok, a.bench_num_prompts, edges, a.seed,
                                                  chat=not a.bench_raw,
                                                  prefix_tokens=a.bench_prefix_tokens)

    n = len(prompts)
    total_in = sum(map(len, prompts))
    print(f"[bench] {n} prompts, {total_in:,} prompt tokens, max_tokens={a.bench_max_tokens}, "
          f"{'ignore_eos' if a.ignore_eos else 'stop at EOS'}, "
          f"shared system prompt of ~{a.bench_prefix_tokens} tokens")
    for name, lo, hi in zip(names, edges, edges[1:]):
        lens = [len(p) for b, p in zip(bins, prompts) if b == name]
        print(f"  {name:6} [{lo:>4}, {hi:>4}) tokens: {len(lens):3d} prompts, "
              f"mean {sum(lens) / len(lens):6.1f} tokens (with system prompt)")

    results = {}

    # 1. the standard way
    print(f"\n[bench] HF generate, static batches of {a.bench_hf_batch_size} ...", flush=True)
    attn = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa"
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=attn).cuda().eval()
    hf_static_generate(hf, tok, prompts[:2], 4, 2, stop_ids, a.ignore_eos)   # warm up
    t = time.perf_counter()
    n_out, done_at, _, _ = hf_static_generate(
        hf, tok, prompts, a.bench_max_tokens, a.bench_hf_batch_size, stop_ids, a.ignore_eos)
    results["HF static"] = (time.perf_counter() - t, n_out, done_at, [0] * n)
    del hf
    gc.collect(); torch.cuda.empty_cache()

    # 2. our way, without and then with prefix caching
    for caching in (False, True):
        name = f"tinyserve ({'prefix cache' if caching else 'no prefix cache'})"
        print(f"[bench] {name} ...", flush=True)
        cfg = Config(model=a.model, attention_backend=a.attention_backend,
                     max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                     max_num_batched_tokens=a.max_num_batched_tokens,
                     gpu_memory_utilization=a.gpu_memory_utilization,
                     use_cuda_graphs=not a.no_cuda_graphs,
                     enable_prefix_caching=caching)
        engine = Engine(cfg)

        # warm up on an unrelated prompt, so the shared system prompt is not
        # already sitting in the cache when the clock starts
        engine.generate([tok.encode("Warm up.")] * 2, SamplingParams(temperature=0.0, max_tokens=4))
        t = time.perf_counter()
        n_out, done_at, cached = tinyserve_generate(engine, prompts, a.bench_max_tokens,
                                                    a.ignore_eos)
        results[name] = (time.perf_counter() - t, n_out, done_at, cached)
        del engine
        gc.collect(); torch.cuda.empty_cache()

    # 3. report
    df = pd.DataFrame({
        name: {
            "wall s": wall,
            "req/s": n / wall,
            "out tok/s": sum(n_out) / wall,
            "all tok/s": (total_in + sum(n_out)) / wall,
            "p50 done s": np.percentile(done_at, 50),
            "p90 done s": np.percentile(done_at, 90),
            "cached %": 100 * sum(cached) / total_in,
        }
        for name, (wall, n_out, done_at, cached) in results.items()
    }).T
    print()
    print(df.round(2).to_string())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["serve", "generate", "kv-demo", "attn-demo", "bench"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--attention-backend", default="flash_attn", choices=["flash_attn", "torch"])
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--bench-num-prompts", type=int, default=100)
    ap.add_argument("--bench-bins", default="1,64,512,2048",
                    help="prompt length bin edges in tokens, prompts are split equally across bins")
    ap.add_argument("--bench-max-tokens", type=int, default=512)
    ap.add_argument("--bench-hf-batch-size", type=int, default=16)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="bench: every request generates exactly --bench-max-tokens")
    ap.add_argument("--bench-prefix-tokens", type=int, default=256,
                    help="bench: length of the system prompt shared by every request (0 = none)")
    ap.add_argument("--bench-raw", action="store_true",
                    help="bench: use the raw instruction text instead of the chat template")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.command == "kv-demo":       # pure bookkeeping: no model, no GPU
        kv_cache_demo()
        return
    if a.command == "attn-demo":     # needs a GPU, but no model weights
        attention_demo()
        return
    if a.command == "bench":         # tinyserve vs HF static batching, offline
        benchmark(a)
        return

    cfg = Config(model=a.model, attention_backend=a.attention_backend,
                 max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                 max_num_batched_tokens=a.max_num_batched_tokens,
                 gpu_memory_utilization=a.gpu_memory_utilization,
                 use_cuda_graphs=not a.no_cuda_graphs,
                 enable_prefix_caching=not a.no_prefix_caching)
    engine = Engine(cfg)
    if a.command == "generate":
        out = engine.generate([a.prompt], SamplingParams(temperature=a.temperature,
                                                         max_tokens=a.max_tokens))[0]
        print(a.prompt + out.text)
        return
    
    import uvicorn
    print(f"[tinyserve] serving {a.model} on http://localhost:{a.port}")
    uvicorn.run(build_app(AsyncEngine(engine), a.model), host="0.0.0.0", port=a.port,
                log_level="warning")


if __name__ == "__main__":
    main()