"""
tinyserve: a small vLLM-style inference engine in one file.

This is a simpler implementation of Paged KV Cache with prefix caching, 
continuous batching with chunked prefill, FlashAttention (or a pure torch reference), 
CUDA graphs for decoding and an OpenAI-compatible HTTP Server

Current support for Llama class models like Llama3, Qwen2/3 and Qwen3MoE
"""
import argparse
import asyncio
import atexit
import copy
import datetime
import glob
import json
import os
import queue
import re
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
import traceback
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers import AutoTokenizer, GenerationConfig

from fastapi import Request as HTTPRequest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except:
    FLASH_ATTN_AVAILABLE = False

try:
    import tinykernels as tk
    TRITON_KERNELS_AVAILABLE = True
except Exception:
    TRITON_KERNELS_AVAILABLE = False

### GLOBAL VARIABLES ###
ATTN_FN = None           # the attention backend: ATTENTION_BACKENDS[cfg.attention_backend]
TRITON_KERNELS = True   # layers call the fused kernels in tinykernels.py instead of torch ops
ATTN_META = None         # AttnMetadata for this step, None during the memory profiling run
TP_RANK, TP_SIZE = 0, 1  # By default we use rank 0 with 1 worker.
TP_CPU_GROUP = None      # gloo (CPU) group used to broadcast python objects each step
TP_WORKERS: list = []    # rank 0 only: the worker processes, so /health can notice one dying
COLUMN_PARALLEL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
ROW_PARALLEL = ("o_proj", "down_proj")
SUPPORTED_MODELS = ("llama", "qwen2", "qwen3", "qwen3_moe")  # config.json model_type values Model can load
EXPERT_KEY = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
FUSED_KEY = re.compile(r"(layers\.\d+\.(?:self_attn|mlp))\.(q_proj|k_proj|v_proj|gate_proj|up_proj)\.(weight|bias)")
FUSED_PARTS = {"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), "v_proj": ("qkv_proj", 2),
               "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1)}

@dataclass
class Config:

    model: str
    dtype: torch.dtype = torch.bfloat16 # bfloat16 inference by default
    max_model_len: str = 4096 # max prompt + output we accept
    attention_backend: str = "triton" # triton (ours, tinykernels.py) | flash_attn | torch (reference)
    use_triton_kernels: bool = True # fused RMSNorm, RoPE, SiLU, MoE (False: the plain torch ops)
    block_size: int = 0 # 0 is autoselect for backend
    max_num_seqs: int = 128 # max requests we can have in a batch
    max_num_batched_tokens: int = 2048 # token budget per step
    gpu_memory_utilization: float = 0.85 # total gpu memory allocated
    enable_prefix_caching: bool = True # reuse common prefixes
    use_cuda_graphs: bool = True # capture cuda graphs for decode only
    seed: int = 0 # determinism

    tensor_parallel_size: int = 1 # split dense layers like Megatron
    expert_parallel: bool = False # MoE under tp > 1: whole experts per GPU instead of slicing every expert


    def __post_init__(self):
        # if flash_attn not available dont use
        if self.attention_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            warnings.warn("flash attention not available, default to torch")
            self.attention_backend = "torch"

        if (self.attention_backend == "triton" or self.use_triton_kernels) and not TRITON_KERNELS_AVAILABLE:
            warnings.warn("triton kernels not available, falling back to flash_attn (or torch) and torch ops")
            if self.attention_backend == "triton":
                self.attention_backend = "flash_attn" if FLASH_ATTN_AVAILABLE else "torch"
            self.use_triton_kernels = False

        # set page size as small as possible based on backend
        if self.block_size == 0:
            self.block_size = 256 if self.attention_backend == "flash_attn" else 16

        # our manually looped torch implementation cant be cuda graphed
        if self.attention_backend == "torch":
            self.use_cuda_graphs = False  # its Python loop can't be captured

        assert self.tensor_parallel_size <= torch.cuda.device_count(), \
            f"--tensor-parallel-size {self.tensor_parallel_size} but only {torch.cuda.device_count()} GPUs visible"

@dataclass
class SamplingParams:
    temperature: float = 1.0 # 0 means greedy decode
    top_p: float = 1.0
    top_k: int = -1          # -1 means off
    max_tokens: int = 64     # max tokens to gen
    repetition_penalty: float = 1.1 # >1 makes recently seen tokens less likely, 1.0 = off
    repeat_last_n: int = 64  # how many recent tokens the penalty looks at, 0 = off, -1 = whole context
    ignore_eos: bool = False # normally we stop gen on eos token
    stop: list[str] | None = None  # allows us to define chars to immediately stop generation

    ### for testing ###
    forced_token_ids: list[int] | None = None # teacher forcing to compare log probs
    return_logprobs: bool = False

class Request:

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

        # async scheduling. We will run the scheduler for step n+1 while the GPU
        # is running step n. This means the new token from step n hasnt arrived yet
        # so we keep a placeholder. the pending_row tells us which row in that
        # steps sampled tokens it is in
        self.pending = 0
        self.pending_row = -1

        # this requests page table (what blocks contain the KV cache
        # for this specific request?)
        self.block_ids: list[int] = []


        # One has per full block, for prefix reuse
        self.block_hashes: list[int] = [] 
        self.num_cached_tokens = 0 # prefix-cache hits at admission
        self.logprobs: list[torch.Tensor] = []
        self.finished = False
        self.finish_reason: str | None = None

        # some metadata about the request if it was preempted or chunk prefilled
        self.was_preempted = False
        self.was_chunked_prefill = False

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def output_ids(self):
        # everything after the prompt is what was generated
        return self.token_ids[self.prompt_len:]

class KVCacheManager:

    def __init__(self, 
                 num_blocks: int, 
                 block_size: int, 
                 enable_caching: bool):

        self.block_size = block_size
        self.enable_caching = enable_caching
        self.num_blocks = num_blocks

        # therefore block 2 cannot be evicted while other requests are still using it
        self.ref_cnt = [0] * num_blocks

        # if a physical block contains a cached prefix, what prefxi does it represent
        self.block_hash : list[int | None] = [None] * num_blocks

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
        # (a placeholder token's value isn't known yet, so its block can't be hashed until it is)
        while len(req.block_hashes) < (req.num_tokens - req.pending) // bs:

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
        # only blocks whose token values are all known (no placeholder); a block skipped
        # here gets registered on a later allocate, once its placeholder is filled in
        for i in range(min(num_tokens, req.num_tokens - req.pending) // self.block_size):

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
        return self.num_computed + self.num_new >= self.req.num_tokens

class Scheduler:

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
                victim.was_preempted = True

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
    slot_mapping: torch.Tensor          # [T]
    query_start_loc: torch.Tensor       # [B+1]
    cu_seqlens_k: torch.Tensor          # [B+1]
    block_table: torch.Tensor           # [B, max_blocks]
    num_seqs: int
    max_query_len: int
    max_seq_len: int
    query_start_loc_cpu: list[int]      # [B+1]
    seq_lens_cpu: list[int] # [B]

def torch_ref_attention(
        q,         # [T ]
        k_cache, 
        v_cache, 
        meta,
        scale
):
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
    return flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q=meta.query_start_loc, cu_seqlens_k=meta.cu_seqlens_k,
        max_seqlen_q=meta.max_query_len, max_seqlen_k=meta.max_seq_len,
        softmax_scale=scale, causal=True, block_table=meta.block_table
    )

def triton_attention(q, k_cache, v_cache, meta, scale):
    return tk.paged_attention(q, k_cache, v_cache, meta, scale)

def shard_config(cfg, tp: int, rank: int, expert_parallel: bool = False):
    """
    The per-GPU model is the same architecture with 1/tp of the attention
    heads, KV heads and MLP width, so the model code needs no changes at all.
    MoE models also get 1/tp of every expert's width (TP-MoE), or with
    expert_parallel, all the width of 1/tp of the experts (see SparseMoE)
    """
    H, Hkv, I = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size
    assert H % tp == 0 and Hkv % tp == 0 and I % tp == 0, \
        f"heads {H}, kv heads {Hkv}, intermediate {I} must all divide by tensor parallel size {tp}"

    ### Create a copy of the config ###
    cfg = copy.deepcopy(cfg)

    ### Update the config ###
    cfg.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // H  # keep the full model's head size
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size = H // tp, Hkv // tp, I // tp

    ### MoE: either slice every expert (like the dense MLP) or hand out whole experts ###
    E = getattr(cfg, "num_experts", 0) or 0
    if E and expert_parallel:
        assert E % tp == 0, f"{E} experts must divide by tensor parallel size {tp}"
        cfg.local_experts = (rank * E // tp, E // tp)   # (first global expert id, how many)
    elif E:
        assert cfg.moe_intermediate_size % tp == 0, \
            f"moe_intermediate_size {cfg.moe_intermediate_size} must divide by tensor parallel size {tp}"
        cfg.moe_intermediate_size //= tp

    return cfg

def load_shard(fh, key: str, rank: int, tp: int) -> torch.Tensor:
    """
    read only this rank's slice of a checkpoint tensor (get_slice avoids
    loading the whole thing). Weights are [out_features, in_features]
    
    fh comes from the weight in 
    
    with safe_open("model.safetensors", framework="pt", device="cpu") as fh:
        weight = load_shard(fh, "model.layers.0.self_attn.q_proj.weight", rank=0, tp=2)

    which runs on CPU. But when loading into the GPU we only want to load the shard
    we care about or we risk OOM
    """

    ### If we just have 1 gpu then nothing to really do here, return ###
    if tp == 1:
        # .get_tensor() loads the entire tensor into a PyTorch tensor.
        return fh.get_tensor(key)

    ### get_slice() lets us read a only a part of the tensors (it is a lazy slide handle) ###
    sl = fh.get_slice(key)

    ### Get the shape of that tensor ###
    shape = sl.get_shape()

    ### If it is column prallel we split along dimension 0 ###
    ### basically our weights are (M x N), if tp=2, we have two separate (M/2 x N) ###
    ### and this is splitting on the output dim of our matmul (as we transpose before matmul)
    if any(p in key for p in COLUMN_PARALLEL):          # weight and bias: split outputs
        n = shape[0] // tp
        return sl[rank * n:(rank + 1) * n]
    
    ### If we are row parallel we split on the second dimension ###
    if any(p in key for p in ROW_PARALLEL):

        ### Bias is added after the all reduce, the bug here would be if every worker added the 
        ### bias. If we have two workers we would add the bias twice! So instead we just ensure that
        ### only rank 0 adds the bias, everything else just adds 0
        if key.endswith("bias"):                        
            t = sl[:]
            return t if rank == 0 else torch.zeros_like(t)

        ### Otherwise slice and return 
        n = shape[1] // tp                             
        return sl[:, rank * n:(rank + 1) * n]

    ### all other parameters are replicated
    return sl[:]                                         

def tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """
    Sum row-parallel partial outputs across GPUs.

    Worker 0 computes y0 = x0 @ W0.
    Worker 1 computes y1 = x1 @ W1.

    All-reduce sums the partial outputs:
        y = y0 + y1

    Both workers receive the complete output y.
    """
    if TP_SIZE > 1:
        dist.all_reduce(x)
    return x

class RMSNorm(nn.Module):
    """x / rms(x) * weight, computed in fp32"""
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float() # float32
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) \
                                 + self.eps) * self.weight.float()).to(x.dtype) # round to bf16
    
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

        ### q, k and v in ONE matmul (QKVParallelLinear from vLLM)
        self.qkv_sizes = [self.H * self.D, self.Hkv * self.D, self.Hkv * self.D]
        self.qkv_proj = nn.Linear(cfg.hidden_size, sum(self.qkv_sizes), bias=qkv_bias)
        self.o_proj = nn.Linear(self.H * self.D, cfg.hidden_size, bias=False)

        self.qk_norm = cfg.model_type in ("qwen3", "qwen3_moe")  # Qwen3 normalizes q and k per head
        if self.qk_norm:
            self.q_norm = RMSNorm(self.D, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.D, cfg.rms_norm_eps)

        self.layer_idx = layer_idx
        self.kv_cache = None  # [2, num_blocks, block_size, Hkv, D]

    def forward(self, x, cos, sin):
        return self.output(self.attend(*self.project(x, cos, sin)))

    def project(self, x, cos, sin):
        """Everything before the paged attention: q/k/v projection, q/k norm, RoPE"""
        T = x.shape[0] # x is passed in as packed

        # one GEMM, then views into its output (no copies, the kernels take strides)
        q, k, v = self.qkv_proj(x).split(self.qkv_sizes, dim=-1)
        q = q.view(T, self.H, self.D)
        k = k.view(T, self.Hkv, self.D)
        v = v.view(T, self.Hkv, self.D)

        ### Optionally use triton kernels for fused ops ###
        if TRITON_KERNELS:
            # q/k norm and both rotary embeddings in one kernel, in place
            tk.qk_norm_rope(q, k, cos, sin,
                            self.q_norm.weight if self.qk_norm else None,
                            self.k_norm.weight if self.qk_norm else None,
                            self.q_norm.eps if self.qk_norm else 0.0)
        else:
            if self.qk_norm:
                q, k = self.q_norm(q), self.k_norm(k)

            q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        return q, k, v

    def output(self, out):
        """Everything after the paged attention: o_proj (+ the tensor-parallel all-reduce)"""
        # row parallel: under tensor parallelism each GPU only has its own heads' share
        return tp_all_reduce(self.o_proj(out.reshape(out.shape[0], -1)))

    def attend(self, q, k, v):
        """
        The paged part of attention:

            1) write this steps new K/V into the page pool
            2) attend over each requests context
        """
        meta = ATTN_META # read from global

        if meta is None:  # profiling run: shapes only
            return q * 0

        # Write this step's new K/V into the page pool before attention
        # (slot_mapping has one slot per token, padding tokens go to the null page 0)
        k_cache, v_cache = self.kv_cache[0], self.kv_cache[1]
        k_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, k)
        v_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, v)

        return ATTN_FN(q, k_cache, v_cache, meta, self.D ** -0.5)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up_sizes = [cfg.intermediate_size, cfg.intermediate_size]
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        if TRITON_KERNELS:
            return tp_all_reduce(self.down_proj(tk.silu_and_mul(gate, up)))
        return tp_all_reduce(self.down_proj(F.silu(gate) * up))

def is_moe_layer(cfg, i: int) -> bool:
    """
    Return whether decoder layer i uses an MoE feed-forward network.

    if config has num_experts not 0, if layer i is not an mlp (dense) layer
    and does this layer fall on the MoE schedule (for example every 3rd layer
    might be MoE). we do i+1 as layer indices start at 0
    """
    return bool(getattr(cfg, "num_experts", 0)) \
        and i not in (getattr(cfg, "mlp_only_layers", None) or []) \
        and (i + 1) % getattr(cfg, "decoder_sparse_step", 1) == 0

def moe_reference(x, w13, w2, topk_weights, topk_ids):
    """
    Reference MoE: Loop over local experts, process the tokens
    routed to each expert, and accumulate their weighted outputs. 

    main issue is nonzero() that produces a data-dependent GPU-to-host
    sync, making it unusable for cuda graph capture, the fused triton
    implementation solves this
    """

    # x: [num_tokens, hidden_size]
    # w13: [num_experts, 2 * intermediate_size, hidden_size]
    # w2: [num_experts, hidden_size, intermediate_size]
    # topk_weights: [num_tokens, top_k]
    # topk_ids: [num_tokens, top_k]
    #
    # topk_ids[t, k] identifies the expert selected for token t at routing
    # topk_ids = torch.tensor([
    #     [0, 2],  # token 0 selects experts 0 and 2
    #     [1, 0],  # token 1 selects experts 1 and 0
    #     [2, 1],  # token 2 selects experts 2 and 1
    # ])

    # topk_weights = torch.tensor([
    #     [0.3, 0.7],  
    #     [0.1, 0.9],  
    #     [0.4, 0.6],  
    # ])

    # on e = expert 2:
    # tok = [0, 2]       # which tokens were assigned to expert 2?
    # slot = [1, 0]      # of the topk, which of the k was it?

    ### Create empty output to accumulate expert results into
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

    ### get the intermediate size
    I = w2.shape[-1]

    ### For every expert
    for e in range(w13.shape[0]):

        ### find every token and top-k slot assignment routed to expert e. 
        ### tok cotains the token index (which token), slot is if the current expert
        ### is assigned to that token. If some token is not assigned to 
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)

        ### If empty, no tokens were assigned to this expert, skip
        if tok.numel() == 0:
            continue

        ### Otherwise index out only the tokens active for this expert
        ### and matmul with the w13[e].t() -> (T x H) @ (H x 2I)
        h = x[tok] @ w13[e].t()                      

        ### Swiglu on the two halves -> (T x I)
        h = F.silu(h[:, :I]) * h[:, I:]

        ### (h @ w2[e].t()).float() -> (T x I) @ (I x H) = (T x H)
        ### * topk_weights[tok, slot, None] -> index out the assigned weight for this expert and mul
        ### out.index_add_(0, tok...) -> in this token position add the weighted output in the final out
        out.index_add_(0, tok, (h @ w2[e].t()).float() * topk_weights[tok, slot, None])

    return out.to(x.dtype)

class SparseMoE(nn.Module):
    """
    Replaces the MLP in MoE layers. A small router scores every expert for every token, 
    the top_k experts run, and their outputs are summed weighted by their router probs

    The checkpoint stores each expert seperately like (expert.5.gate_proj.weight, ...)
    so we stack them into one and our kernel can index any expert by an offset

        w13: [E_local, 2 * I, H]   gate rows, then up rows
        w2:  [E_local, H, I]

    Under tensor parallel there are two ways to split the experts:

        TP-MoE: Every GPU holds all E experts, but each cut to I/tp of its
                width just like a dense MLP. This way we get perfect load 
                balance across GPUs
        
        Expert Parallel: GPU holds whole experts, so if we have 32 experts
                         and 2 gpus, each holds 16 experts. Each gpu has
                         its copy of the full input hidden state, so each 
                         runs its own experts on the tokens routed to them
                         and writes zero for the rest. (i.e. if token 8
                         is assigned to expert 7 and 14, and GPU0 holds experts
                         0-15 and GPU1 holds 16-31, then GPU0 will do the weighted
                         sum of the expert outputs and GPU1 will just set 0 for that token. 
                         On the other hand if the experts are 7 and 20, then
                         GPU0 will run expert 7 and multiply by weight, GPU1 will
                         will run expert 20 and multiply by weight, at the end we
                         sum the contributions from different GPUs and that does
                         the full accumulation!)

    """
    def __init__(self, cfg):
        super().__init__()

        ### Grab configuration shape
        E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.top_k = cfg.num_experts_per_tok
        self.norm_topk_prob = getattr(cfg, "norm_topk_prob", True) # apply softmax to topk router logits

        ### the router is tiny, every GPU keeps a full copy and routes every token ###
        self.gate = nn.Linear(H, E, bias=False)

        ### which experts live here: all of them, unless shard_config said otherwise (EP) ###
        self.first_expert, self.num_local = getattr(cfg, "local_experts", (0, E))
        self.expert_parallel = self.num_local < E
        self.w13 = nn.Parameter(torch.empty(self.num_local, 2 * I, H))
        self.w2 = nn.Parameter(torch.empty(self.num_local, H, I))

        ### global expert id -> local id on this GPU, or -1 if another GPU owns it ###
        expert_map = torch.full((E,), -1, dtype=torch.long)
        expert_map[self.first_expert:self.first_expert + self.num_local] = torch.arange(self.num_local)
        self.register_buffer("expert_map", expert_map, persistent=False)

    def load_expert(self, expert: int, proj: str, fh, key: str, rank: int, tp: int) -> bool:
        """copy one checkpoint tensor (experts.{expert}.{proj}.weight) into the stacked weights"""
        e = expert - self.first_expert
        if not 0 <= e < self.num_local:
            return False                          # another GPU's expert, don't even read it

        ### EP keeps whole experts. TP-MoE slices gate/up by rows and down by columns ###
        ### exactly like the dense MLP, which load_shard does by name ###
        w = fh.get_tensor(key) if self.expert_parallel else load_shard(fh, key, rank, tp)

        ### What is the input shape
        I = self.w2.shape[-1]

        ### Copy in both halves of the gate and up
        if proj == "gate_proj":
            self.w13[e, :I].copy_(w)
        elif proj == "up_proj":
            self.w13[e, I:].copy_(w)
        else:
            ### copy the out proj
            self.w2[e].copy_(w)

        return True

    def forward(self, x):
        ### route: softmax over all experts, keep the top_k, optionally renormalize them ###
        probs = self.gate(x).float().softmax(-1)
        topk_weights, topk_ids = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

        ### global ids -> local ids (-1 for other GPUs' experts under EP) ###
        topk_ids = self.expert_map[topk_ids]

        ### pick between options
        moe = tk.fused_moe if TRITON_KERNELS else moe_reference

        ### all reduce between GPUs
        return tp_all_reduce(moe(x, self.w13, self.w2, topk_weights, topk_ids))

class DecoderLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.self_attn = Attention(cfg, i)
        self.mlp = SparseMoE(cfg) if is_moe_layer(cfg, i) else MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    """
    The residual stream travels as two tensors, (x, residual), the stream being
    residual + x: the add is folded into the next norm. Why, from the fused kernels:

        Our RMSNorm triton kernel optionally allows us to fuse the residual connection. What 
        that means is we see that self.post_attention_layernorm for example applies to 

            norm2(residual + attn_out)

        Now this would incur two kernel launches, one for the sum, one for the norm. A cheaper
        fused op would be:

            residual: B x T x E
            attn_out: B x T x E

            each triton thread loads both a row of residual and a row of attn_out, do the sum
            right there and then norm that row, basically the idea of fusing our operations!
 
        Now above the norm operation above:


            LAYER 1:

            input: x0
            norm1: rmsnorm1(input)
            attn_out: attention(norm1)
            residual1: input + attn_out

            norm2: rmsnorm2(residual1)
            mlp_out: mlp(norm2)
            residual2: residual1 + mlp_out

            LAYER 2:
                        
            input: residual2
            norm1: rmsnorm1(input)
            attn_out: attention(norm1)
            residual1: residual2 + attn_out

            norm2: rmsnorm2(residual1)
            mlp_out: mlp(norm2)
            residual2: residual1 + mlp_out


        So we can easily fuse:

            residual1: x0 + attn_out
            norm2: rmsnorm2(residual1)

            into

            residual1, norm2 = fusedrmsnorm2(x0, attn_out)

            
        The challenge is rmsnorm1 in the next layer. Notice that the input to rnsnorm1 in layer 2 is
        residual2, which was residual1 + mlp_out. To fuse it in the rms norm we must
        then pass both the residual1 AND the mlp_out from the previous transformer block
        to the next and then compute:

        LAYER 2:
                                
        input: residual1, mlp_out
        residual_2, norm1 = fusedrmsnorm1(residual1, mlp_out)

    The plain torch path uses the same (x, residual) convention, just with an
    ordinary add before each norm (add_rmsnorm does one or the other)
    """

    def forward(self, x, residual, cos, sin):
        """(x, residual) -> (x, residual) for the next layer. residual=None for the first layer"""
        h, residual = add_rmsnorm(self.input_layernorm, x, residual)
        h, residual = add_rmsnorm(self.post_attention_layernorm, self.self_attn(h, cos, sin), residual)
        return self.mlp(h), residual

def add_rmsnorm(norm: RMSNorm, x, residual):
    """
    residual += x, then norm(residual): one kernel with the Triton kernels on.
    Returns (normed, residual). residual=None (the first layer): nothing to add yet
    """
    if TRITON_KERNELS:
        if residual is None:
            return tk.rmsnorm(x, norm.weight, norm.eps), x
        return tk.rmsnorm(x, norm.weight, norm.eps, residual=residual), residual
    residual = x if residual is None else residual + x
    return norm(residual), residual

class Model(nn.Module):
    """Llama / Qwen2 / Qwen3 / Qwen3-MoE (the model_types in SUPPORTED_MODELS)"""

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
        x, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            x, residual = layer(x, residual, cos, sin)
        return add_rmsnorm(self.norm, x, residual)[0]     # the last residual add + final norm

    def logits(self, hidden):
        ### separate method to get logis
        w = self.embed_tokens.weight if self.tie else self.lm_head.weight
        return hidden @ w.t()

def load_model(name: str, dtype, max_position: int, tp_rank: int = 0, tp_size: int = 1,
               expert_parallel: bool = False):
    """quick helper method to load weights into our model (this rank's shard of them)"""
    path = name if os.path.isdir(name) else snapshot_download(
        name, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    cfg = AutoConfig.from_pretrained(path)
    assert cfg.model_type in SUPPORTED_MODELS, \
        f"model_type {cfg.model_type!r} isn't supported, these are: {SUPPORTED_MODELS}"
    if tp_size > 1:
        cfg = shard_config(cfg, tp_size, tp_rank, expert_parallel)
    torch.set_default_dtype(dtype)
    with torch.device("cuda"):
        model = Model(cfg, max_position)
    torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    loaded, unused = set(), []
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, "pt", device=f"cuda:{torch.cuda.current_device()}") as fh:
                for key in fh.keys():
                    name_ = key.removeprefix("model.")

                    ### MoE: separate tensors per expert go into the layer's stacked w13 / w2 ###
                    if m := EXPERT_KEY.fullmatch(name_):
                        moe = model.layers[int(m[1])].mlp
                        if moe.load_expert(int(m[2]), m[3], fh, key, tp_rank, tp_size):
                            loaded.add(f"layers.{m[1]}.mlp.{'w2' if m[3] == 'down_proj' else 'w13'}")

                    ### q/k/v and gate/up: each goes into its row range of the fused projection.
                    ### load_shard still sees the original name, so under tensor parallelism it
                    ### returns THIS GPU's slice of q (or k, ...), and each GPU's fused weight
                    ### ends up [its q heads; its k heads; its v heads], never a cut of the stack
                    elif m := FUSED_KEY.fullmatch(name_):
                        module = model.get_submodule(m[1])
                        fused, part = FUSED_PARTS[m[2]]
                        sizes = module.qkv_sizes if fused == "qkv_proj" else module.gate_up_sizes
                        start = sum(sizes[:part])
                        getattr(getattr(module, fused), m[3])[start:start + sizes[part]].copy_(
                            load_shard(fh, key, tp_rank, tp_size))
                        loaded.add(f"{m[1]}.{fused}.{m[3]}")
                    elif name_ in params:
                        params[name_].copy_(load_shard(fh, key, tp_rank, tp_size))
                        loaded.add(name_)
                    else:
                        unused.append(key)

    ### a weight the checkpoint never filled would silently stay random, so fail loudly ###
    ### (usually a new architecture whose names don't match ours) ###
    missing = sorted(set(params) - loaded)
    assert not missing, f"{len(missing)} parameters not found in the checkpoint, e.g. {missing[:3]}"

    ### the other direction: checkpoint tensors we have no place for. Harmless for a ###
    ### tied lm_head, but e.g. Qwen1.5-MoE's shared_expert would be silently dropped ###
    if unused and tp_rank == 0:
        warnings.warn(f"ignored {len(unused)} checkpoint tensors, e.g. {unused[:3]}")
    return model.eval()

def to_gpu(data, dtype=None) -> torch.Tensor:
    """
    Host data (a list or numpy array) -> a GPU tensor, Nonblocking
    """
    t = torch.as_tensor(data, dtype=dtype)
    return t.pin_memory().to("cuda", non_blocking=True)

def apply_repetition_penalty(logits, params: list[SamplingParams], token_ids: list[list[int]]):
    """
    Sometimes LLMs seem to generate the same token over and over! So it is typical to apply
    a repetition penalty, where every time a token occurs in the last repeat_last_n tokens
    of the context, it gets its logits nudged towards being less likely, by dividing by the
    penalty is positive, or multiplying if negative. 

    Ollama / llama.cpp style repeat penalty. Every token that appears in the
    last repeat_last_n tokens of the context (prompt included) gets its logit
    pushed toward "less likely": divided by the penalty if positive, multiplied
    if negative. Once per distinct token, however many times it repeated.

        penalty 1.1, logits [2.0, -1.0, 0.5], token 0 and 1 seen recently
            -> [2.0 / 1.1, -1.0 * 1.1, 0.5] = [1.82, -1.1, 0.5]

    This is basically OLLAMA repetition penalty, one caveat is there is no difference between
    a token occuring once or 10 times in the last N tokens, that token gets the same repetition
    penalty. This is why vLLM has other options like presense penalty where we subtract a fixed
    amount from every token that has appeared once, or frequency penalty where we subtract
    an amount proportional the the token's occurance count

    Edits logits IN PLACE (only the R * n penalized entries, never a copy of the whole
    [B, V] tensor). sample() hands us a copy when a request wants the raw logprobs.
    """
    ### logits come in as (B x VOCAB_SIZE)
    ### token_ids is a B length list of lists, of all tokens that have come so far in every request 
    rows, windows = [], []

    ### for each request and its params
    for i, (p, ids) in enumerate(zip(params, token_ids)):

        ### if we have a repetition penalty over a non-0 last n tokens
        if p.repetition_penalty != 1.0 and p.repeat_last_n != 0:

            ### append which request we are on
            rows.append(i)

            ### get all the ids in the previous repeat_last_n 
            windows.append(ids if p.repeat_last_n < 0 else ids[-p.repeat_last_n:])

    ### if no request needs penalty just return 
    if not rows:
        return logits

    # PyTorch tensors must be rectangular, but requests can have
    # repetition windows of different lengths.
    # Pad every window to the length of the longest window by repeating
    # its first token. This keeps the padding ID valid.
    n = max(len(w) for w in windows)

    # Shape: [R, n], where R is the number of active requests.
    #
    # Example:
    #   [10, 20, 30] -> [10, 20, 30, 10]
    #   [7, 8]       -> [7, 8, 7, 7]
    #
    # Repeated IDs are harmless here: gather reads the same original logit
    # for each occurrence, so scatter writes the same adjusted value to it.
    # (via numpy: torch.tensor on a nested list converts element by element in
    # Python, ~1 ms for 256 x 64 ids; np.array does it in C, then one copy)
    ids = to_gpu(np.array([w + [w[0]] * (n - len(w)) for w in windows], dtype=np.int64))   # [R, n]

    # One penalty per active request.
    # [:, None] changes [R] into [R, 1], allowing each request's penalty
    # to broadcast across all n token IDs in that request's window.
    penalty = to_gpu([params[i].repetition_penalty for i in rows], torch.float32)[:, None] # [R x 1]

    # The batch row of each penalized request, as [R, 1] so it broadcasts
    # against ids [R, n]: logits[rows, ids] then addresses exactly the R * n
    # (row, token) pairs we care about, without touching the rest of the vocab
    rows = to_gpu(rows, torch.long)[:, None]

    # For each request, read the logits at the token IDs in its window
    #   logits: [B, V]  -> all vocabulary logits
    #   seen:   [R, n]  -> logits[rows[r], ids[r, j]]
    seen = logits[rows, ids]

    # Adjust and write back in place:
    #   positive logits are divided by the penalty;
    #   zero or negative logits are multiplied by the penalty.
    #
    # Duplicate IDs may target the same position multiple times, but
    # each occurrence has the same adjusted value because all values
    # were read before the write.
    logits[rows, ids] = torch.where(seen > 0, seen / penalty, seen * penalty).to(logits.dtype)

    return logits

def sample(logits, params: list[SamplingParams], token_ids: list[list[int]] | None = None):
    """One batched call for mixed sampling settings.

    logits:    [B, V]  next-token logits, one row per request that produced
                       a real sample this step (completes_prompt).
    params:    length B, per-request temperature / top_k / top_p / repetition penalty.
    token_ids: length B, each request's context so far (for the repetition penalty)

    Returns the sampled token ids as a [B] GPU tensor, NOT a Python list: turning
    them into a list would make the CPU wait for the GPU. Engine.step keeps them on
    the GPU for the next step's inputs and copies them back in the background
    """
    # the penalty goes first (llama.cpp's order), so it also steers greedy decoding.
    # It edits logits in place, so copy them first if a request wants the raw logprobs
    if token_ids is not None:
        if any(p.return_logprobs for p in params):
            logits = logits.clone()
        logits = apply_repetition_penalty(logits, params, token_ids)

    # if temp is 0 we assume greedy decode
    if all(p.temperature == 0 for p in params):
        return logits.argmax(-1)

    # what gpu
    dev = logits.device

    # get vocab size
    V = logits.shape[-1]

    # top_k <= 0 means off -> keep the whole vocab for that row. A greedy row
    # only ever takes its single best token, so it needs just 1 candidate (and
    # mustn't force a full sort on the rest of the batch with its top_k = -1)
    ks = [1 if p.temperature == 0 else (p.top_k if p.top_k > 0 else V) for p in params]
    K = max(ks) # what is the largest k we need to grab for this batch of requests?

    # Every per-row setting in ONE host -> device copy
    settings = to_gpu([[p.temperature or 1.0, k, p.top_p, p.temperature == 0]
                       for p, k in zip(params, ks)], torch.float32)
    temp, top_k, top_p, is_greedy = settings.unbind(1)
    is_greedy = is_greedy.bool()

    # If K==V we have not reduced the vocab size, meaning we are just doing
    # a simple temperature sampling, scale and apply gumbel max
    if K == V and bool((top_p[~is_greedy] >= 1).all()):
        noisy = logits.float() / temp[:, None]
        noisy -= torch.empty_like(noisy).exponential_().log()
        return torch.where(is_greedy, logits.argmax(-1), noisy.argmax(-1))

    # We improve sampling, instea of running .sort() on the full vocab and then
    # select topk, we first filter out topk and then sort
    # top-k runs on the RAW logits: dividing a row by a positive temperature
    # doesn't change its order, so only the K survivors need scaling below,
    # not the whole [B, V] tensor
    if K < V:
        sorted_logits, idx = logits.topk(K, dim=-1)              # [B, K]
    else:
        sorted_logits, idx = logits.sort(-1, descending=True)    # [B, V]

    # On our [B x K], everythign below is scaling, per-row top-k/p and gumbel max
    # so we can make one kernel to fuse together the small launches. although if 
    # K is large (no top-p/k) then it doesnt matter as work and memory traffic
    # increase with K and processing logits dominate the launch overhead
    if TRITON_KERNELS and K <= 1024:
        return tk.sample_from_topk(sorted_logits, idx, temp, top_k, top_p)

    # Scale
    sorted_logits = sorted_logits.float() / temp[:, None] # fp32 from here, higher temp -> flatter

    # ranks 0..K-1, 0 is the largest logit. Rows with a smaller top_k than K
    # drop their extra candidates here
    ranks = torch.arange(K, device=dev)[None, :]
    sorted_logits = sorted_logits.masked_fill(ranks >= top_k[:, None], float("-inf"))

    # top-p on what top-k kept (renormalized, the same order HF and vLLM use):
    # drop a candidate if the ones before it already reach top_p of the mass
    probs = sorted_logits.softmax(-1)
    drop = probs.cumsum(-1) - probs > top_p[:, None]
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

    # Gumbel-max: argmax(logit + Gumbel(0,1)) ~ Categorical(softmax(logits)).
    # Done over the K candidates directly, then mapped back to vocab ids, so
    # nothing vocab-sized is built after the topk
    gumbel = -torch.empty_like(sorted_logits).exponential_().log()
    choice = (sorted_logits + gumbel).argmax(-1, keepdim=True)
    sampled = idx.gather(-1, choice).squeeze(-1)

    # greedy rows: the first top-k candidate IS the argmax, so no extra pass over the vocab to find it
    return torch.where(is_greedy, idx[:, 0], sampled)

@dataclass
class PackedBatch:
    """One step's inputs as plain host data (see ModelRunner._pack)"""
    ids: list[int]           # [T] token ids, all requests packed
    pos: list[int]           # [T] absolute positions
    slots: list[int]         # [T] KV cache slot each new token is written to
    qsl: list[int]           # [B+1] query start offsets
    seq_lens: list[int]      # [B] kv_len per request after this step
    bt: np.ndarray           # [B, max_blocks] block table
    logit_rows: list[int]    # rows of the output we sample from
    max_query_len: int
    fill_rows: list[int]     # input rows holding a placeholder token (see Request.pending) ...
    fill_src: list[int]      # ... and where its real id sits in prev_tokens

class ModelRunner:
    """
    Make the schedulers decisions into GPU tensors and run the model
    """

    def __init__(self, cfg):

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        t = time.time()
        self.model = load_model(
            cfg.model, cfg.dtype, cfg.max_model_len, TP_RANK, TP_SIZE, cfg.expert_parallel
        )
        if TP_RANK == 0:
            print(f"[tinyserve] model weights: {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
                  f"{' per GPU' if TP_SIZE > 1 else ''}, loaded in {time.time() - t:.1f}s")

        # get the model config
        mc = self.model.cfg

        # MoE layers, if any. The reference expert loop syncs with the host for
        # every expert, which a CUDA graph can't record, so it runs eagerly
        self.moe = [l.mlp for l in self.model.layers if isinstance(l.mlp, SparseMoE)]
        if self.moe and not cfg.use_triton_kernels and cfg.use_cuda_graphs:
            if TP_RANK == 0:
                warnings.warn("MoE with --no-triton-kernels uses the python expert loop, "
                              "which can't be CUDA graphed: running decode eagerly")
            cfg.use_cuda_graphs = False

        # Get metadata (per GPU: with tensor parallel these are already divided by tp)
        self.H, self.Hkv = mc.num_attention_heads, mc.num_key_value_heads
        self.D = getattr(mc, "head_dim", None) or mc.hidden_size // self.H
        self.num_layers = mc.num_hidden_layers

        # set global var for attention, and whether layers use the fused kernels
        ATTENTION_BACKENDS = {
            "flash_attn": flash_attention,
            "torch": torch_ref_attention,
            "triton": triton_attention,
        }

        global ATTN_FN, TRITON_KERNELS
        ATTN_FN = ATTENTION_BACKENDS[cfg.attention_backend]
        TRITON_KERNELS = cfg.use_triton_kernels

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

        # Place to store graphs: ("decode", batch size) or ("mixed", token count)
        # -> (graph, its output buffer), see capture_cuda_graphs
        self.graphs: dict[tuple[str, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

        # Last step's sampled token ids. They stay on the GPU (the next step's input
        # ids are copied from here, see _upload) and also go to pinned host memory in
        # the background, where Engine.step picks them up once they've arrived
        self.prev_tokens = z(S, dt=torch.int64)
        self.tokens_host = torch.zeros(S, dtype=torch.int64, pin_memory=True)

    @torch.inference_mode()
    def autotune_triton(self):
        """
        Triton's autotuner benchmarks every candidate config the first time a
        kernel sees a new shape key, then caches the winner. Trigger all of it
        here at startup rather than on real requests (where it would stall
        traffic for seconds) or inside a CUDA graph capture (where the graph
        would record the benchmark runs)
        """
        ### if we are using triton kernels ###
        if self.cfg.use_triton_kernels:
            # MoE first, on random routing (every layer has the same shapes, one is enough)
            if self.moe:
                tk.autotune_fused_moe(self.moe[0].w13, self.moe[0].w2, self.moe[0].top_k,
                                      self.cfg.max_num_batched_tokens)

            # run the worst case scenario to trigger autotuning 
            n = self.cfg.max_num_batched_tokens
            self.model(self.input_ids[:n], self.positions[:n])

        ### if we are using attention backend 
        if self.cfg.attention_backend == "triton":

            ### Trigger autotuning for both prefill and decode
            best = tk.autotune_prefill_attention(self.H, self.Hkv, self.D, self.cfg.block_size,
                                                 self.cfg.dtype, self.cfg.max_num_batched_tokens)
            tk.autotune_decode_attention(self.H, self.Hkv, self.D, self.cfg.block_size,
                                         self.cfg.dtype, self.cfg.max_model_len)
            
            if TP_RANK == 0:
                print(f"[tinyserve] triton attention autotuned (prefill: {best})")

        torch.cuda.synchronize()

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

        # Triton kernels autotune on first use, which allocates benchmarking
        # scratch, so get that out of the way before measuring the peak
        self.autotune_triton()

        # reset stats
        torch.cuda.reset_peak_memory_stats()

        # Measure the high-water mark of weights + activations + workspace
        # for the largest packed step we will ever run.
        n = self.cfg.max_num_batched_tokens
        h = self.model(self.input_ids[:n], self.positions[:n])

        # Logits path sized for a full decode batch (one row per seq slot).
        self.model.logits(h[: self.cfg.max_num_seqs]).float().softmax(-1)   # worst case: fp32 copies of all logits

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

        # Tensor parallel: every rank must use the same page count, since rank 0's
        # one block table addresses all of them. Free memory can differ per GPU
        # so everyone takes the smallest case
        if TP_SIZE > 1:
            t = torch.tensor([num_blocks], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            num_blocks = int(t.item())

        # After reserving page 0, the pool must still hold one max-length sequence.
        assert (num_blocks - 1) * self.cfg.block_size >= self.cfg.max_model_len, \
            "not enough GPU memory for even one max_model_len sequence"

        # One tensor per layer: [K|V, page, offset, kv_head, dim].
        # Same page-id space in every layer, so one block_table serves all of them.
        for layer in self.model.layers:
            layer.self_attn.kv_cache = torch.zeros(
                2, num_blocks, self.cfg.block_size, self.Hkv, self.D,
                dtype=self.cfg.dtype, device="cuda")

        if TP_RANK == 0:
            print(f"[tinyserve] KV cache: {num_blocks} blocks x {self.cfg.block_size} tokens = "
                  f"{num_blocks * self.cfg.block_size:,} tokens "
                  f"({num_blocks * bytes_per_block / 2**30:.1f} GB{' per GPU' if TP_SIZE > 1 else ''})")
            # how many requests fit at once if every one used the full context
            print(f"[tinyserve] maximum concurrency for {self.cfg.max_model_len:,} tokens per request: "
                  f"{(num_blocks - 1) * self.cfg.block_size / self.cfg.max_model_len:.2f}x")
        
        return num_blocks

    def _pack(self, batch: list[ScheduledRequest], pad_to: int = 0, pad_tokens: int = 0) -> "PackedBatch":
        """
        Pack this step's inputs as plain lists / numpy, _upload then copies them into
        the persistent buffers the model/graphs read. Under Tensor Parallelism, this is 
        exactly what rank 0 broadcasts each step, the other ranks never see a Request object

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

        Padding, to match a captured CUDA graph:
            pad_to:      (decode graphs) add dummy 1-token requests up to pad_to requests
            pad_tokens:  (mixed graphs) add token rows that belong to no request up to
                         pad_tokens tokens, and empty requests (q_len 0) up to the
                         graph's request count, mixed_num_seqs(pad_tokens)

        Additionally when preparing this batch, it is actually for step n+1, the actual
        prediction of the next token in step n is currently in flight. So when we append our
        ids, we dont actually have the last one yet... So we give it a placeholder, and then 
        when we actually go to _upload the next step to the model we grab those right then and there
        
        """
        bs = self.cfg.block_size
        num_seqs = max(len(batch), pad_to, self.mixed_num_seqs(pad_tokens) if pad_tokens else 0)

        # packed streams we will copy onto the gpu buffers
        ids, pos, slots, logit_rows = [], [], [], []
        fill_rows, fill_src = [], []
        qsl, seq_lens = [0], [] # qsl starts at 0; we append the rest

        # page table for real + dummy rows (padding rows point at the null page 0)
        bt = np.zeros((num_seqs, self.max_blocks_per_seq), dtype=np.int32)

        # prepare batches (each sample in the batch is one of the ScheduledRequest)
        for i, s in enumerate(batch):
            start, n = s.num_computed, s.num_new          # already-have-KV, this-step queries
            p = np.arange(start, start + n)               # absolute positions = q_pos
            blocks = np.asarray(s.req.block_ids)          # what are these requests block_ids

            # pack together each batch into a single stream
            ids.extend(s.req.token_ids[start:start + n])  # packed input ids
            if s.req.pending and start + n == s.req.num_tokens:
                # the newest token is a placeholder (its value is still on the GPU,
                # sampled last step): put a 0 here, _upload copies the real id in
                ids[-1] = 0
                fill_rows.append(len(ids) - 1)
                fill_src.append(s.req.pending_row)
            # position embeds restart from 0 for every sample
            pos.extend(p.tolist())                        # RoPE / causal positions

            # slot = page * bs + offset, destination of index_copy_ for each new token
            slots.extend((blocks[p // bs] * bs + p % bs).tolist())

            qsl.append(qsl[-1] + n)                       # request i owns q[qsl[i]:qsl[i+1]]
            seq_lens.append(start + n)                    # kv_len after this step's write
            bt[i, : len(blocks)] = blocks                 # page-table row
            logit_rows.append(qsl[-1] - 1)                # last packed row -> sample logits here

        # Dummy decode request so T and B match the captured graph, slot 0 writes
        # to the null page, outputs are never sampled when doing graph capture, we dont
        # pass in data, just padding so all our q_len=1 for all causes, which is the case
        # for decoding anyway
        for _ in range(len(batch), pad_to):
            ids.append(0)
            pos.append(0)
            slots.append(0)
            qsl.append(qsl[-1] + 1)
            seq_lens.append(1)

        # Mixed graphs: the token count and the request count are both the graph's.
        # The extra tokens belong to no request (qsl stops before them): they run
        # through the per-token layers, their K/V goes to the null page, and attention
        # never writes their rows. The extra requests are empty (q_len 0, kv_len 0),
        # so every attention program that lands on one exits right away
        max_query_len = max(s.num_new for s in batch) if batch else 1
        if pad_tokens:
            ids += [0] * (pad_tokens - len(ids))
            pos += [0] * (pad_tokens - len(pos))
            slots += [0] * (pad_tokens - len(slots))
            qsl += [qsl[-1]] * (num_seqs - len(seq_lens))
            seq_lens += [0] * (num_seqs - len(seq_lens))
            max_query_len = pad_tokens   # the graph has to serve prefill chunks: run the prefill launch

        return PackedBatch(ids, pos, slots, qsl, seq_lens, bt, logit_rows,
                           max_query_len, fill_rows, fill_src)

    def _upload(self, p: "PackedBatch", for_graph: bool):
        """Device half of a step's inputs: fill the persistent buffers, build AttnMetadata"""
        T, B = len(p.ids), len(p.seq_lens)

        # persistent buffers: fixed addresses for CUDA graphs
        self.input_ids[:T].copy_(to_gpu(p.ids, torch.int64))
        self.positions[:T].copy_(to_gpu(p.pos, torch.int64))
        self.slot_mapping[:T].copy_(to_gpu(p.slots, torch.int64))
        self.query_start_loc[: B + 1].copy_(to_gpu(p.qsl, torch.int32))
        self.cu_seqlens_k[: B + 1].copy_(to_gpu([0] + np.cumsum(p.seq_lens).tolist(), torch.int32))
        self.block_table[:B].copy_(to_gpu(p.bt))

        # Placeholder tokens: their ids were sampled last step and never left the
        # GPU, copy them into place on the GPU. Under tensor parallelism only rank 0
        # sampled, so it broadcasts them first (every rank sees the same fill_rows)
        if p.fill_rows:
            if TP_SIZE > 1:
                dist.broadcast(self.prev_tokens, src=0)
            rows, src = to_gpu(p.fill_rows, torch.int64), to_gpu(p.fill_src, torch.int64)
            self.input_ids[rows] = self.prev_tokens[src]

        # build the attn metadata
        max_seq = self.cfg.max_model_len if for_graph else max(p.seq_lens)

        meta = AttnMetadata(
            self.slot_mapping[:T],
            self.query_start_loc[: B + 1],
            self.cu_seqlens_k[: B + 1],
            self.block_table[:B],
            B,
            p.max_query_len,
            max_seq,
            p.qsl, p.seq_lens, # host copies for torch_ref
        )

        return T, meta

    @torch.inference_mode()
    def graph_sizes(self):
        # decode graph bucket sizes. Finegrained at bottom (1, 2, 4)
        # where small batches are common, then every 8 upto max_num_seqs, 
        # a live batch of N picks the smallest bucket >= N and pads the
        # rest with dummy 1 token requests on the null page
        sizes = [1, 2, 4] + list(range(8, self.cfg.max_num_seqs + 1, 8))
        return [s for s in sizes if s <= self.cfg.max_num_seqs]

    def mixed_sizes(self):
        # token-count buckets for mixed graphs (steps with a prefill chunk): a step of
        # T tokens runs the smallest bucket >= T. The extra rows are junk nobody reads,
        # but they still cost matmul work, so the buckets are fine-grained (every 16
        # tokens, like vLLM's capture sizes). Past 512 tokens a step is busy enough with
        # compute that kernel launches hardly matter: those run eager.
        # Only our Triton attention can be captured for any mix of requests (its launch
        # shape depends on the token and request counts only, see tinykernels.py):
        # flash_attn's depends on the longest prompt chunk, so it gets decode graphs only
        if self.cfg.attention_backend != "triton":
            return []
        return [s for s in range(16, 513, 16) if s <= self.cfg.max_num_batched_tokens]

    def mixed_num_seqs(self, num_tokens: int) -> int:
        # requests a mixed graph of num_tokens tokens has room for (each request
        # brings at least one token, and there are never more than max_num_seqs)
        return min(num_tokens, self.cfg.max_num_seqs)

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

        Two kinds, both of the whole forward pass:

            ("decode", B):  B requests with one new token each (the common step)
            ("mixed", T):   T tokens, any mix of decodes and prefill chunks, padded
                            to T tokens and mixed_num_seqs(T) requests (see _pack)

        A graph replays its launches with the grid it was captured with, so this
        only works because every kernel's launch shape depends on nothing but those
        counts. Attention is the one that cares how the tokens split into requests,
        and our Triton kernel reads that from the GPU (see tinykernels.paged_attention)
        """
        global ATTN_META
        t = time.time()

        # One workspace pool for every graph. Separate pools would each reserve
        # their own peak activation slab, sharing means we pay about max(peaks) 
        # rather than the sum of all our peaks
        pool = torch.cuda.graph_pool_handle()

        # Largest first, the first capture allocates the worst case scenario pool
        # and smaller graphs attach to that pool instead of growing it
        captures = [("mixed", T) for T in reversed(self.mixed_sizes())] + \
                   [("decode", B) for B in reversed(self.graph_sizes())]
        for kind, size in captures:

            # No real requests, just the padding: dummy decode rows (q_len=1, slot 0),
            # or for mixed, token rows of no request and empty requests.
            # for_graph=True bakes max_seq_len = max_model_len into metadata.
            packed = self._pack([], pad_to=size) if kind == "decode" else self._pack([], pad_tokens=size)
            T, ATTN_META = self._upload(packed, for_graph=True)

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
            self.graphs[(kind, size)] = (g, out)

        if TP_RANK == 0:
            n_mixed = len(self.mixed_sizes())
            print(f"[tinyserve] captured {len(self.graphs) - n_mixed} decode + {n_mixed} mixed "
                  f"CUDA graphs in {time.time() - t:.1f}s")

    def prepare(self, batch: list[ScheduledRequest]):
        """
        CPU half of a step: graph or eager, then pack the inputs into plain lists.
        No GPU work, so Engine.step runs it while the GPU is busy with the step before
        """
        # Decode graph: every request has one new token, at the smallest batch bucket
        # that fits. Real rows first, dummy decode rows after. Dummies write slot 0
        # (null page) and are not in logit_rows.
        is_decode = all(s.num_new == 1 for s in batch)
        size = next((s for s in self.graph_sizes() if s >= len(batch)), None)
        if self.graphs and is_decode and size is not None:
            return self._pack(batch, pad_to=size), "decode", size

        # anything else (a prefill chunk in the batch): a mixed graph at the smallest
        # token-count bucket that fits, or eager past the largest one
        T = sum(s.num_new for s in batch)
        size = next((s for s in self.mixed_sizes() if s >= T), None)
        if self.graphs and size is not None:
            return self._pack(batch, pad_tokens=size), "mixed", size
    
        # Otherwise run it eagerly
        return self._pack(batch), "eager", None

    @torch.inference_mode()
    def launch_forward(self, prepared) -> torch.Tensor:
        """
        Queue one step's forward pass on the GPU and return its logits. Returns as
        soon as the work is queued, the logits are still being computed
        """
        packed, mode, size = prepared

        # tensor parallel: the other GPUs must run this exact step too
        if TP_SIZE > 1:
            tp_broadcast(prepared)

        hidden = self.forward_step(packed, mode, size)

        # hidden is  packed [T, ...], logit_rows indexes the last query of
        # each real request, and graph padding is skipped. 
        return self.model.logits(hidden[to_gpu(packed.logit_rows, torch.int64)])

    @torch.inference_mode()
    def launch_sample(self, batch: list[ScheduledRequest], logits: torch.Tensor) -> torch.cuda.Event:
        """
        Queue sampling. The token ids stay on the GPU in prev_tokens (the next
        step's inputs come from there) and are copied to tokens_host in the
        background. Returns an event that fires once that copy has landed
        """
        
        # Sample the tokens
        tokens = sample(logits, [s.req.params for s in batch], [s.req.token_ids for s in batch])
        B = len(batch)

        # Copy the tokens into our CPU and GPU buffers
        self.prev_tokens[:B].copy_(tokens)
        self.tokens_host[:B].copy_(tokens, non_blocking=True)

        # Start an event to capture later that triggers once its done!
        done = torch.cuda.Event()
        done.record()

        return done

    @torch.inference_mode()
    def forward_step(self, packed: PackedBatch, mode: str, size: int | None) -> torch.Tensor:
        """
        The forward pass for one step
        """
        global ATTN_META
        if mode != "eager":
            T, ATTN_META = self._upload(packed, for_graph=True)
            graph, out = self.graphs[(mode, size)]
            graph.replay()                 # one launch, no allocs, same addresses
            return out                     # the buffer captured above
        
        T, ATTN_META = self._upload(packed, for_graph=False)   # true T, true max(seq_lens)
        return self.model(self.input_ids[:T], self.positions[:T])

def tp_worker(rank: int, cfg: Config, port: int):
    """
    Entrypoint for all Non-Zero ranks (the target of spawn_tp_workers). Each worker
    owns its own local model shard and KV cache, rank 0 handles requests, scheduling,
    and sampling. Compare with Engine.__init__, which does steps 2 and 3 the same way.
    """

    ### join the process group (blocks until every rank, including rank 0 has arrived) ###
    init_tensor_parallel(rank, cfg.tensor_parallel_size, port)

    ### same setup as rank 0, meeting it at every collective inside ###
    ### num_blocks is not needed, rank 0 sends finished block tables every step ###
    runner, _ = build_runner(cfg)

    ### wait for rank 0 to broadcast the next inference step, run it, repeat ###
    while (step := tp_broadcast()) is not None:
        runner.forward_step(*step)

    ### stop_tensor_parallel() on rank 0 broadcast None. Release this process's ###
    ### own resources, the same way rank 0 does ###
    release_tensor_parallel(runner)

def init_tensor_parallel(rank: int, world_size: int, port: int):

    """
    This method connects each process to the same distributed communication groups, 
    every process runs this method, rank 0 calls it from Engine.__init__ while ranks
    1...N-1 call it inside tp_worker(). The idea is that all ranks will join teh same 
    NCCL and GLOO Groups. 

    """

    ### These are process local globals, Each spawned pythin process has its own copy, so
    ### each rank records its own rank and world size
    global TP_RANK, TP_SIZE, TP_CPU_GROUP

    ### Select this processes GPU, this assumes rank 0 uses gpu 0, rank 1 uses gpu 1, etc... ###
    torch.cuda.set_device(rank)

    ### Join the default distributed process group. 
    ###
    ### Each rank uses the same rendezvous address, worldsize and a unique rank. Initialization
    ### coordinates the ranks, a rank taht arrives early will wait for other to join, the backend NCCL
    ### Is used for GPU-to-GPU comm such as all-reduce. Device-id eagerly initializes the NCCL communication
    ### for this GPU specifically
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            world_size=world_size, rank=rank, device_id=torch.device(f"cuda:{rank}"))
    
    ### This is for CPU side coordination that doesnt need GPU tensor communication ###
    ### for example rank 0 can tell other workers that the next decoding step should process ###
    ### a partial batch of requests ###
    TP_CPU_GROUP = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=365))

    ### Set this workers global vars for later ###
    TP_RANK, TP_SIZE = rank, world_size

def build_runner(cfg: Config) -> tuple[ModelRunner, int]:
    """
    Load the mdoel, allocate KV cache, Capture CUDA Graphs. EVery rank runs this, 
    rank 0 from the Engine, and otehrs from tp_worker(). Allocate KV Cache and
    capture_cuda_graphs are collectives, an operation that all ranks have to call
    before any of them can be finished with it. For example when rank 0 calls all_reduce()
    it cant do anything until rank 1 also calls all_reduce(), We just have to ensure the
    order of operations line up between ranks so nobody is waiting for someone else. 
    """
    ### Local: load only this rank's shard of the weights ###
    runner = ModelRunner(cfg)

    ### Sync: each rank sizes the cache from its own free memory, then all_reduce MIN ###
    ### so everyone gets the same page count (rank 0's block tables index every GPU) ###
    num_blocks = runner.allocate_kv_cache()

    ### Sync: the warm-up and captured forwards contain the o_proj/down_proj all_reduces ###
    if cfg.use_cuda_graphs:
        runner.capture_cuda_graphs()

    return runner, num_blocks

def tp_broadcast(obj=None):
    """
    Broadcast an inference step object from rank 0 to every rank in TP_CPU_GROUP. 

    Rank 0 (sender):
        Calls to_broadcast(obj) with the inference step it prepared
    
    Other Ranks (receivers):
        calls tp_broadcast() with no argument, their local obj defaults to None

    All ranks call the same broadcast collective. Because src=0, PyTorch takes the
    object from rank 0 and replaces each recievers local box[0] with that object. The
    function then returns box[0] on every rank
    
    """

    ### Each process creates its own local list. Rank 0's list contains the
    ### object to send, receivers lists initially contain None
    box = [obj]

    ### Rank 0 supplies the data, all other ranks recieve it
    ### this uses the CPU/Gloo to process group to broadcast the python object
    dist.broadcast_object_list(box, src=0, group=TP_CPU_GROUP)

    ### rank 0 just gets its original object, but the recieves get the object
    ### from rank 0!
    return box[0]

def spawn_tp_workers(cfg: Config) -> int:
    """
    This is the thing that acutally calles the tp_worker from earlier

    Workers are launched with the spawn start method, each worker starts its own
    python interpreter and reimports the launching module, so code that starts
    the engine has to be protected by if __name__ == "__main__" or they would all
    spawn their own which would be bad
    """
    ### Find an available TCP port for the distributed process to use when establishing
    ### its process groups.
    with socket.socket() as s:

        ### this binds the socket to localhost, and 0 is synonymous for "find any random
        ### available port and assign it to me"
        s.bind(("127.0.0.1", 0))

        ### Returns a tuple containing ("127.0.0.1", RAND_PORT_NUMBER)
        port = s.getsockname()[1] # extract that port number

    # use spawn so each worker starts in a new python interpreter
    ctx = mp.get_context("spawn")

    ### Launch all nonzero ranks, each child starts tp_worker(), which calls
    ### init_tensor_parallel() for its own rank
    for rank in range(1, cfg.tensor_parallel_size):

        ### Start a worker process for this rank, Daemon workers are terminated
        ### when the parent process exits
        p = ctx.Process(target=tp_worker, args=(rank, cfg, port), daemon=True)
        p.start()

        ### Keep the process handle so the server can monitor worker health and inspect
        ### their status
        TP_WORKERS.append(p)

    return port

def release_tensor_parallel(runner: ModelRunner):
    """
    Step 5, on every rank. Free this process's CUDA graphs, then its distributed
    communication resources (NCCL communicator, gloo sockets).
    """
    # the graphs recorded NCCL all-reduces: while they exist the communicator
    # counts as in use and destroy_process_group waits forever
    runner.graphs.clear()
    torch.cuda.synchronize()
    dist.destroy_process_group()

def stop_tensor_parallel(runner: ModelRunner):
    """
    Shutdown tensor parallelism when the main process exits
    Engine.__init__ registers this with atexit.

    Rank 0 broadcasts None as a shutdown signal, Worker processes receive
    None from tp_broadcast() causing their inference loops to terminate.
    Each worker then releases its own distributed process setup

    Rank 0 also releases its own here.
    """
    ### Tell everyone None
    tp_broadcast(None)

    ### Release rank 0
    release_tensor_parallel(runner)

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
    logprobs: list[torch.Tensor] | None = None  
    was_preempted: bool = False
    was_chunked_prefill: bool = False

class Detokenizer:
    """
    Incrementally convert generated token IDs into text.
    We also avoid emitting a trailing U+FFFD ("�") because it can mean that
    a multi-byte UTF-8 character is incomplete and needs another token.
    """

    def __init__(self, tok):
        self.tok = tok
        self.text = ""
        self.prefix = 0
        self.read = 0
        self.released = 0

    def update(self, ids, final) -> str:
        old = self.tok.decode(
            ids[self.prefix:self.read],
            skip_special_tokens=True,
        )

        new = self.tok.decode(
            ids[self.prefix:],
            skip_special_tokens=True,
        )

        if len(new) <= len(old) or (new.endswith("�") and not final):
            return ""

        delta = new[len(old):]
        self.text += delta
        self.prefix, self.read = self.read, len(ids)

        return delta

def apply_stop_strings(text: str, stop: list[str]) -> tuple[str, bool]:
    """
    Filter generated text against stop strings, preserving streaming safety. 

    Returns (safe_text, hit) where:
        safe_text: the prefix safe to emit to the client, excluding any stop string and 
                   any trailing partial matches that could become one
        hit: True if a complete stop string was found, otherwise false
    
    If a complete stop string occurs in text, return the text before its earliest occurance
    and True. If no complete stop is found, withold the longest suffix of text that matches a 
    proper prefix of any stop string, returning the remaining prefix and False

    This prevents partial stop strings leaking to the client when a stop sequence spans multiple
    generation chunks, the caller should pass cumulative generated text, emit only safe_text, and 
    stop generation when hit is True

    Example with stop=["Observation:"]:
    text = "x = 1\nObserv"       -> ("x = 1\n", False)
    text = "x = 1\nObservation:" -> ("x = 1\n", True)
    text = "hello world"         -> ("hello world", False)
    """
    # Find the earliest occurrence of any complete stop string.
    # text.find(s) returns its starting index, or -1 if it is absent.
    cut = min((i for s in stop if (i := text.find(s)) != -1), default=-1)
    if cut != -1:
        return text[:cut], True
    
    # No complete stop string was found.
    # Check whether the end of text matches the beginning of any stop string.
    # If so, those trailing characters might become a stop string when more
    # text arrives, so they must be withheld from the client.
    #
    # For each stop string s, test every proper, nonempty prefix s[:k].
    # Keep the length of the longest prefix that matches the end of text.
    hold = max((k for s in stop for k in range(1, len(s)) if text.endswith(s[:k])), default=0)
    return text[:len(text) - hold], False

class Engine:
    """
    Put it all together! 

    Engine by default launches on GPU 0 (rank 0) only, from which we startup all the other workers
    if we have tp > 1
    """
    def __init__(self, cfg: Config):

        t_start = time.time()
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

        ### Tensor Parallel: this process is rank 0 (GPU 0). The steps below line up with ###
        ### tp_worker(), see TENSOR PARALLEL LIFECYCLE above spawn_tp_workers ###
        if cfg.tensor_parallel_size > 1:

            ### start ranks 1...N-1, each immediately blocks in init_tensor_parallel ###
            port = spawn_tp_workers(cfg)

            ### rank 0 joins, the rendezvous completes and every rank moves on ###
            init_tensor_parallel(0, cfg.tensor_parallel_size, port)

        ### model, KV cache, CUDA graphs. With tp > 1 the workers run this same ###
        ### function at the same time and meet us at the collectives inside ###
        self.runner, num_blocks = build_runner(cfg)

        ### On a normal python shutdown, tell the workers to stop ###
        if cfg.tensor_parallel_size > 1:
            atexit.register(stop_tensor_parallel, self.runner)

        ### (rank 0 only): the CPU-side bookkeeping. Workers don't need any, rank 0 ###
        ### sends them finished block tables with every step (see ModelRunner.run) ###
        self.kv = KVCacheManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg, self.kv)

        # set up storage for the rest
        self.detok: dict[str, Detokenizer] = {}
        self.requests: dict[str, Request] = {}

        # a few counters purely for /metrics 
        self.tokens_last_step = 0

        # async scheduling (see step): the batch scheduled + packed for the next
        # step, and the step whose sampled tokens are still on their way back
        self.next_step = None
        self.in_flight = None
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0

        # running totals for log_stats while serving
        self.num_prompt_tokens_run = 0      # prompt tokens prefilled (chunks, recomputes)
        self.num_generated_tokens = 0       # tokens sampled for requests
        self._last_log = (time.time(), 0, 0)

        print(f"[tinyserve] engine ready in {time.time() - t_start:.1f}s")

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
        return self.scheduler.has_work() or self.next_step is not None or self.in_flight is not None

    def maybe_log_stats(self, interval: float):
        """log_stats() if interval seconds passed and something happened since the last line"""
        t, prompt0, gen0 = self._last_log
        if time.time() - t < interval:
            return
        if self.has_work() or (self.num_prompt_tokens_run, self.num_generated_tokens) != (prompt0, gen0):
            self.log_stats()
        else:
            self._last_log = (time.time(), prompt0, gen0)   # idle: start the next window quietly

    def log_stats(self):
        """
        serve status over time to print out
        """
        now = time.time()
        t, prompt0, gen0 = self._last_log
        dt = max(now - t, 1e-6)
        self._last_log = (now, self.num_prompt_tokens_run, self.num_generated_tokens)
        kv_used = 1 - self.kv.num_free / (self.kv.num_blocks - 1)       # page 0 is the null page
        hit = self.total_cached_tokens / max(self.total_prompt_tokens, 1)
        print(f"[tinyserve] prompt throughput: {(self.num_prompt_tokens_run - prompt0) / dt:.1f} tok/s, "
              f"generation throughput: {(self.num_generated_tokens - gen0) / dt:.1f} tok/s, "
              f"running: {len(self.scheduler.running)} reqs, waiting: {len(self.scheduler.waiting)} reqs, "
              f"KV cache usage: {kv_used:.1%}, prefix cache hit rate: {hit:.1%}", flush=True)

    def step(self):
        """
        The CPU stays one step ahead by preparing the next batch before the current
        batch has even finished sampleing. This works because our scheduling needs
        to know how many tokens each request will process NOT their actual token ids

            1. GPU: Run the forward pass for step i, using the batch prepped in the
                    previous step i-1
            2. CPU: Process i-1s sampled tokens while the GPU runs step i, Update request
                    histories, check stopping conditions, and prepare outputs
            3. GPU: Sample tokens from i's logits, keep the sampled IDS on the GPU for
                    the next forward pass, and copy them to pinned CPU memory for request
                    management
            4. CPU: Schedule and pack step i+1 without waiting for step i's sampeld token
                    values. Insert placehoders for the unkown IDs, the next upload
                    replaces them with the actual GPU-side tokens

        Tradeoffs:
            - Outputs are processed by the CPU one step after they are sampled
            - A request that finishes during step i may already be included in step 
              i+1's batch, causaing some wasted compute. Its row is discarded once the 
              CPU actually processes the stopping condition
        """
        # 1. Launch the forward pass for the batch prepared in the previous step, this
        #    submits GPU work and returns without waiting for it to finish as PyTorch GPU
        #    ops are async w.r.t to the CPU (demos/pytorch_async.py)
        batch, logits = None, None
        if self.next_step is not None:
            batch, prepared = self.next_step
            logits = self.runner.launch_forward(prepared)
            self.next_step = None

        # 2. Finish the previous step while the current forward pass runs
        #    wait for its sampled tokens, fill placeholders, update request_state
        #    check stopping conditions and collect complete outputs
        outputs = self._finish_step(*self.in_flight) if self.in_flight is not None else []
        self.in_flight = None

        # 3. Sample tokens for the current batch and prepare requests for the next step. Step 2
        #    has resolved all the previous steps placeholders, so the token histories contain
        #    real IDs for repetition-penalty sampling. Requests that finish in step 2 may
        #    have already run this forward pass and their sampled rows will be discarded
        if batch:
            done = self.runner.launch_sample(batch, logits)

            # Count prompt tokens processed in this batch for statistics.
            self.num_prompt_tokens_run += sum(s.num_new for s in batch if s.num_computed < s.req.prompt_len)

            live = []
            for i, s in enumerate(batch):
                req = s.req

                # A partial prompt chunk writes K/V into cache, but its last query position is not the 
                # end of the prompt its sampled tokens are not valid so we discard
                if not s.completes_prompt:
                    req.was_chunked_prefill = True
                    continue

                # The request may have finished while this forward pass was running, dont schedule more work
                if req.finished:
                    continue

                # keep the index of this requests sampled row for _finish_step
                live.append(i)

                # Teacher forcing (equivalence tests). ignore the sample and
                # append a predetermined id so two engines walk the same path.
                # this is just for debugging. Its value is known now, no placeholder
                forced = req.params.forced_token_ids
                if forced is not None:
                    req.token_ids.append(forced[len(req.output_ids)])
                else:
                    # otherwise use a placeholder
                    req.token_ids.append(-1)    # placeholder, filled in by _finish_step
                    req.pending, req.pending_row = 1, i

                # The length limits only need the count: stop scheduling the request
                # now (its last token is still on its way, _finish_step sends it out)
                if len(req.output_ids) >= req.params.max_tokens or req.num_tokens >= self.cfg.max_model_len:
                    self.scheduler.finish(req, "length")

            self.in_flight = (batch, live, logits, done)
            self.num_generated_tokens += len(live)

        # 4. schedule + pack the next step while the GPU is still busy
        nxt = self.scheduler.schedule()

        # tokens that step spends out of max_num_batched_tokens -
        # a prefill chunk counts its whole chunk, a decode row counts 1
        self.tokens_last_step = sum(s.num_new for s in nxt) if nxt else 0
        if nxt:
            self.next_step = (nxt, self.runner.prepare(nxt))
        return outputs

    def _finish_step(self, batch, live, logits, done):
        """
        Finish processing the previous step's sampled tokens while the GPU
        works on the next batch.

        The CPU and GPU overlap their work. The GPU can start the next forward
        pass before the CPU has finished processing the previous step's output,
        because the sampled token IDs are kept in GPU memory for reuse.

        Timeline:

            1. GPU samples tokens for step i, launch_sample() stores the sampled ids
               immediately in the prev_tokens on GPU, but async copy into tokens_host
               for the CPU. This means the GPU path has access to the next token up front
               but the CPU may not yet as we have a GPU -> CPU comm
            2. CPU doesnt care, itll go ahead and just start prepping step i+1 without
               waiting for tokens_host to populate. It knows how many tokens to expect
               already so we populate with placeholder IDs into the request histories
               and records which sampled GPU row will replace each placeholder
            3. The GPU starts the forward pass for i+1, during upload, _upload() replaces
               the dummy input ids with the actual sampled ids that are already sitting in 
               the GPU prev_tokens. There is no need to wait on the CPU for this
            4. The CPU finsihes step i, those tokens have finally arrived over the GPU-CPU comm
               into the tokens_host, the CPU can now safely read the sampled IDs, replace the 
               placeholders in the request histories, check for stopping conditions, detokenize
               and construct outputs for the serving side
            5. Meanwhile, the GPU may continue executing step i+1, after its forward pass
               it samples the next tokens which are retained on the GPU and copied
               async to the CPU for the next call to _finish_step()
        """

        # launch sample started a cuda stream, we force a sync point as we need the sampled
        # token ids from the previous step to finish processing the requests and the GPU-CPU
        # comm may not be done yet.  So we wait until the previous steps async copy to cpu is
        # complete!
        done.synchronize()                
        tokens = self.runner.tokens_host[:len(batch)].tolist()

        outputs = []
        for i in live:
            req = batch[i].req

            # the reqst may have been cancelled while its GPU work was in flight
            # we just ignore the results if its already been removed from the active requests
            if req.request_id not in self.requests:
                continue

            # Replace the placeholder appended in step 3 of step() with the actual sampled token
            # clear pending so its ready for the next step
            if req.pending:
                req.token_ids[-1] = tokens[i]
                req.pending = 0
            tok = req.token_ids[-1]

            # optionally save teh full next-token log probs distribution, copying
            # the logits to CPU syncs with GPU work, this is mostly for debugging
            if req.params.return_logprobs:
                req.logprobs.append(logits[i].float().log_softmax(-1).cpu())

            # Handle EOS or other stop-tokens. A request may already be marked
            # finished due to its length limit in step 3, if its final token is EOS
            # report "stop", otherwise finish it now which releases KV pages
            if tok in self.stop_ids and not req.params.ignore_eos:
                if req.finished:
                    req.finish_reason = "stop"
                else:
                    self.scheduler.finish(req, "stop")

            # incremental detokenization
            d = self.detok[req.request_id]
            delta = d.update(req.output_ids, req.finished)
            text = d.text

            # stop strings: finish the moment one appears (freeing its KV pages
            # right away), and only ever release text that can't be part of one
            if req.params.stop:
                text, hit = apply_stop_strings(d.text, req.params.stop)
                if hit and not req.finished:
                    self.scheduler.finish(req, "stop")
                elif req.finished and not hit:
                    text = d.text  # ended some other way: flush the held-back tail
                delta, d.released = text[d.released:], len(text)

            # Store the output
            outputs.append(Output(
                req.request_id, text, delta, req.output_ids, req.finished,
                req.finish_reason, req.prompt_len, req.num_cached_tokens,
                req.logprobs if req.params.return_logprobs else None,
                req.was_preempted, req.was_chunked_prefill,
            ))

            # If done clear it out
            if req.finished:
                self.total_prompt_tokens += req.prompt_len
                self.total_cached_tokens += req.num_cached_tokens
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

class AsyncEngine:
    """
    This class bridges the async web server (FastAPI) with a synchronous, compute heavy 
    model engine. It runs the model in a dedicated background thread, so that web requests
    are never blocked while the GPU is processing.
    """
    def __init__(self, engine: Engine, log_interval: float = 10.0):

        self.engine = engine

        # print engine.log_stats() every log_interval seconds while busy (0 = never)
        self.log_interval = log_interval

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
                    # (while idle, still wake up once per log interval to print
                    # the stats of the burst that just ended)
                    kind, rid, payload = self.inbox.get(block=block, timeout=(self.log_interval or None) if block else None)
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

            # vLLM-style status line every log_interval seconds while there's activity
            if self.log_interval:
                self.engine.maybe_log_stats(self.log_interval)

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
        room = eng.cfg.max_model_len - prompt_len
        max_tokens = get(body, "max_completion_tokens", get(body, "max_tokens", room))
        stop = body.get("stop")  # a string or a list of strings
        return SamplingParams(
            temperature=get(body, "temperature", defaults["temperature"]),
            top_p=get(body, "top_p", defaults["top_p"]),
            top_k=get(body, "top_k", defaults["top_k"]),
            # vLLM's name, or Ollama's (repeat_penalty)
            repetition_penalty=get(body, "repetition_penalty",
                                   get(body, "repeat_penalty", SamplingParams.repetition_penalty)),
            repeat_last_n=get(body, "repeat_last_n", SamplingParams.repeat_last_n),
            max_tokens=min(max_tokens, room),
            ignore_eos=get(body, "ignore_eos", False),
            stop=[stop] if isinstance(stop, str) else (list(stop) if stop else None),
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

        def notes(o: Output) -> dict:
            # non-standard fields, purely for observability (benchmarks/simulate_serving.py's
            # dashboard reads these via the openai SDK's model_extra, which
            # preserves unrecognized top-level fields rather than dropping them):
            # was this request ever preempted to free pages for an older one,
            # and did its own prompt need more than one step to prefill
            return {"tinyserve_preempted": o.was_preempted,
                    "tinyserve_chunked_prefill": o.was_chunked_prefill}

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
                        chunk.update(notes(o))
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
            **notes(final),
        })

    # Utility endpoint for load balancers to check if server is alive
    @app.get("/health")
    async def health():
        # under tensor parallelism a dead worker means every request would hang
        dead = [p.pid for p in TP_WORKERS if not p.is_alive()]
        if dead:
            return JSONResponse({"status": "error", "message": f"tensor-parallel worker(s) {dead} died"},
                                status_code=503)
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
        kv = engine.engine.kv
        eng = engine.engine
        return {
            "running": len(s.running),
            "waiting": len(s.waiting),
            "free_kv_blocks": kv.num_free,
            "total_kv_blocks": kv.num_blocks,
            "tokens_last_step": eng.tokens_last_step,
            "max_num_batched_tokens": eng.cfg.max_num_batched_tokens,
            "total_prompt_tokens": eng.total_prompt_tokens,
            "total_cached_tokens": eng.total_cached_tokens,
        }

    # Debug/demo endpoint: which request owns each physical KV block right
    # now, for visualizing the cache (see benchmarks/simulate_serving.py --dashboard).
    # Walks every block on every call - fine for a demo-scale pool, not
    # meant to be polled against a many-hundred-thousand-block one.
    @app.get("/metrics/blocks")
    async def metrics_blocks():
        kv = engine.engine.kv
        s = engine.engine.scheduler

        max_blocks = 20000
        if kv.num_blocks > max_blocks:
            return {"too_many_blocks": True, "num_blocks": kv.num_blocks}

        # a block held by more than one running request right now is a
        # prefix-cache hit actively in progress, not just sitting cached
        owner_count: dict[int, int] = {}
        owner_of: dict[int, str] = {}
        for req in s.running:
            for b in req.block_ids:
                owner_count[b] = owner_count.get(b, 0) + 1
                owner_of[b] = req.request_id

        owners = sorted(set(owner_of.values()))
        owner_index = {rid: i for i, rid in enumerate(owners)}
        block_hash = kv.block_hash

        cells = []
        for b in range(kv.num_blocks):
            if owner_count.get(b, 0) > 1:
                cells.append(-3)  # shared across >1 live request: cache hit in progress
            elif b in owner_of:
                cells.append(owner_index[owner_of[b]])  # owned by exactly one request
            elif block_hash is not None and block_hash[b] is not None:
                cells.append(-2)  # idle but cached: reusable without recompute
            else:
                cells.append(-1)  # free

        return {"num_blocks": kv.num_blocks, "block_size": kv.block_size,
                "owners": owners, "cells": cells}

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["serve", "generate"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=SamplingParams.repetition_penalty,
                    help="generate: >1 discourages repeating recent tokens, 1.0 = off")
    ap.add_argument("--repeat-last-n", type=int, default=SamplingParams.repeat_last_n,
                    help="generate: how many recent tokens the penalty looks at (0 = off, -1 = all)")
    ap.add_argument("--attention-backend", default="triton", choices=["triton", "flash_attn", "torch"],
                    help="triton: our paged attention kernel (tinykernels.py), torch: the slow reference")
    ap.add_argument("--no-triton-kernels", action="store_true",
                    help="use the plain torch ops instead of the fused Triton RMSNorm / RoPE / SwiGLU / MoE kernels")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                    help="split the model across this many GPUs")
    ap.add_argument("--expert-parallel", "-ep", action="store_true",
                    help="MoE models with -tp > 1: give each GPU whole experts instead of a slice of every expert")
    ap.add_argument("--log-stats-interval", type=float, default=10.0,
                    help="serve: print throughput / queue / KV cache stats every this many seconds (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = Config(model=a.model, attention_backend=a.attention_backend,
                 use_triton_kernels=not a.no_triton_kernels,
                 max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                 max_num_batched_tokens=a.max_num_batched_tokens,
                 gpu_memory_utilization=a.gpu_memory_utilization,
                 use_cuda_graphs=not a.no_cuda_graphs,
                 enable_prefix_caching=not a.no_prefix_caching,
                 tensor_parallel_size=a.tensor_parallel_size,
                 expert_parallel=a.expert_parallel)
    engine = Engine(cfg)
    if a.command == "generate":
        out = engine.generate([a.prompt], SamplingParams(temperature=a.temperature,
                                                         max_tokens=a.max_tokens,
                                                         repetition_penalty=a.repetition_penalty,
                                                         repeat_last_n=a.repeat_last_n))[0]
        print(a.prompt + out.text)
        return
    
    import uvicorn
    print(f"[tinyserve] serving {a.model} on http://localhost:{a.port}")
    uvicorn.run(build_app(AsyncEngine(engine, a.log_stats_interval), a.model), host="0.0.0.0", port=a.port,
                log_level="warning")


if __name__ == "__main__":
    main()