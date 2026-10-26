"""
tinykernels: optional Triton kernels for tinyserve.
"""
import functools
from types import SimpleNamespace
import torch
import triton
import triton.language as tl

_PREFILL_CONFIGS = [
    triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
    for m in (64, 128) for n in (32, 64) for w in (4, 8) for s in (2, 3)
]
_DECODE_CONFIGS = [  # BLOCK_M is fixed for decoding
    triton.Config({"BLOCK_N": n}, num_warps=w, num_stages=s)
    for n in (32, 64, 128) for w in (4, 8) for s in (2, 3)
]

@triton.jit
def _rms_rstd(sum_sq, n, eps):
    """
    1 / rms(x) from x's sum of squares over its n elements.
    """
    return tl.rsqrt(sum_sq / n + eps)

@triton.autotune(configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4, 8, 16)],
                 key=["N", "HAS_RESIDUAL"], restore_value=["res_ptr"])
@triton.jit
def _rmsnorm_kernel(
    x_ptr, # [N x E] 
    res_ptr, # [N x E]
    w_ptr,   # [E]
    out_ptr,  
    x_stride, 
    res_stride, 
    out_stride,
    N, 
    eps,
    HAS_RESIDUAL: tl.constexpr, 
    BLOCK: tl.constexpr
):
    
    # one program per row, the whole row lives in registers
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)

    if HAS_RESIDUAL:
        # update our residual stream with residual = residual + x
        r = tl.load(res_ptr + row * res_stride + cols, mask=mask, other=0.0)
        x = (x.to(tl.float32) + r.to(tl.float32)).to(r.dtype)
        tl.store(res_ptr + row * res_stride + cols, x, mask=mask)

    ### Perform the standard RMSNorm in float32 and cast back
    xf = x.to(tl.float32)
    rstd = _rms_rstd(tl.sum(xf * xf, axis=0), N, eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * out_stride + cols,
             (xf * rstd * w).to(out_ptr.dtype.element_ty), mask=mask)

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float,
            residual: torch.Tensor | None = None):
    """
    RMSNorm with optional fused residual addition

    If residual is None, compute:
        out = rmsnorm(x) * weight

    If residual is provided, compute the following in a single triton kernel:
        residual += x
        out = rmsnorm(residual) * weight

    """

    ### Reshape to make [N x E] (though in our packed setup its already this technically) 
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N)

    # allocate output
    out = torch.empty_like(x2)

    # If a residual is provided, flatten it into the same row layout.
    # Otherwise, use x itself as the input to RMSNorm, the kernel skips
    # the residual addition when HAS_RESIDUAL=False so its just placeholder
    r2 = residual.view(-1, N) if residual is not None else x2

    # Run kernel
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_kernel[(x2.shape[0],)](
        x2, 
        r2, 
        weight, 
        out,
        x2.stride(0), 
        r2.stride(0), 
        out.stride(0),
        N, 
        eps,
        HAS_RESIDUAL=residual is not None, BLOCK=BLOCK,
    )

    return out.view(shape)

@triton.jit
def _norm_rope_heads(
    x_ptr,      # [T x H x D]
    tok_stride, 
    head_stride, 
    w_ptr, 
    cos, 
    sin, 
    t, 
    n_heads, 
    eps,
    HAS_NORM: tl.constexpr, 
    HP: tl.constexpr,       # HP is padded number of heads (next power of 2)
    HALF: tl.constexpr
):
    
    # Process all heads for one token. Each head has D dimensions, split
    # into two halves beacuse rotate-half RoPE pairs x[i] with x[i+D/2]. 
    # the resulting tiles have shape [HP, D/2]

    heads = tl.arange(0, HP)[:, None] # Indexes for number of heads [HP x 1]
    d = tl.arange(0, HALF)[None, :] # indexes from 0 -> D/2 [1 x D/2]
    mask = heads < n_heads # dont load padded heads
    base = x_ptr + t * tok_stride + heads * head_stride # index over to the right T and then the right head

    # Load the first and second half [x1 | x2]
    x1 = tl.load(base + d, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF + d, mask=mask, other=0.0).to(tl.float32)

    if HAS_NORM:
        # Qwen3 RMS-normalizes each head of q and k before RoPE
        # same formula as the rmsnorm kernel, the head's sum of squares is just
        # split across its two halves
        ss = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)
        rstd = _rms_rstd(ss, 2 * HALF, eps)[:, None]
        w1 = tl.load(w_ptr + d).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
        # the reference RMSNorm hands bf16 to rotary, round the same way
        x1 = (x1 * rstd * w1).to(x_ptr.dtype.element_ty).to(tl.float32)
        x2 = (x2 * rstd * w2).to(x_ptr.dtype.element_ty).to(tl.float32)

    # [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    tl.store(base + d, (x1 * cos - x2 * sin).to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(base + HALF + d, (x2 * cos + x1 * sin).to(x_ptr.dtype.element_ty), mask=mask)

@triton.autotune(configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4, 8)],
                 key=["n_q_heads", "n_k_heads", "HALF", "HAS_NORM"], restore_value=["q_ptr", "k_ptr"])
@triton.jit
def _qk_norm_rope_kernel(
    q_ptr, 
    k_ptr, 
    cos_ptr, 
    sin_ptr, 
    qw_ptr, 
    kw_ptr,
    q_tok_stride, 
    q_head_stride, 
    k_tok_stride, 
    k_head_stride, 
    cs_stride,
    n_q_heads, 
    n_k_heads, 
    eps,
    HAS_NORM: tl.constexpr, 
    QP: tl.constexpr, 
    KP: tl.constexpr,
    HALF: tl.constexpr
):
    # one program per token: its cos/sin row is loaded once and shared by
    # every q head and every k head
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, HALF)[None, :]
    cos = tl.load(cos_ptr + t * cs_stride + d)
    sin = tl.load(sin_ptr + t * cs_stride + d)
    _norm_rope_heads(q_ptr, q_tok_stride, q_head_stride, qw_ptr, 
                     cos, sin, t, n_q_heads, eps, HAS_NORM, QP, HALF)
    _norm_rope_heads(k_ptr, k_tok_stride, k_head_stride, kw_ptr, 
                     cos, sin, t, n_k_heads, eps, HAS_NORM, KP, HALF)

def qk_norm_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 q_weight: torch.Tensor | None = None, k_weight: torch.Tensor | None = None,
                 eps: float = 1e-6):
    """
    In place on q [T, H, D] and k [T, Hkv, D], optional per-head RMSNorm
    (Qwen3's q_norm / k_norm), then rotate-half RoPE with cos/sin [T, D/2].
    """
    T, H, D = q.shape
    assert D % 2 == 0 and triton.next_power_of_2(D // 2) == D // 2, "head_dim/2 must be a power of 2"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and cos.stride(-1) == 1 and sin.stride(-1) == 1
    has_norm = q_weight is not None

    _qk_norm_rope_kernel[(T,)](
        q, 
        k, 
        cos, 
        sin,
        q_weight if has_norm else q, 
        k_weight if has_norm else k,
        q.stride(0), 
        q.stride(1), 
        k.stride(0), 
        k.stride(1), 
        cos.stride(0),
        H, 
        k.shape[1], 
        eps,
        HAS_NORM=has_norm, 
        QP=triton.next_power_of_2(H),
        KP=triton.next_power_of_2(k.shape[1]), 
        HALF=D // 2,
    )

    return q, k

@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w)
                          for b in (512, 1024, 2048) for w in (2, 4, 8)], key=["N"])
@triton.jit
def _silu_mul_kernel(
    a_ptr, 
    b_ptr, 
    out_ptr, 
    a_stride, 
    b_stride, 
    out_stride, 
    N,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(a_ptr + row * a_stride + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * b_stride + cols, mask=mask, other=0.0).to(tl.float32)
    # F.silu(gate) returns bf16 before the multiply, so round there too
    s = (a * tl.sigmoid(a)).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row * out_stride + cols, (s * b).to(out_ptr.dtype.element_ty), mask=mask)

def silu_and_mul(gate: torch.Tensor, up: torch.Tensor):
    """silu(gate) * up in one pass instead of three kernels and two temporaries."""
    shape = gate.shape
    N = shape[-1]
    a, b = gate.reshape(-1, N), up.reshape(-1, N)
    out = torch.empty_like(a)
    grid = lambda META: (a.shape[0], triton.cdiv(N, META["BLOCK"]))
    _silu_mul_kernel[grid](a, b, out, a.stride(0), b.stride(0), out.stride(0), N)
    return out.view(shape)

@triton.jit
def _attn_kv_tile(
    acc,            # [BLOCK_M x D] running output, not yet divided by the softmax sum (fp32)
    m_i,            # [BLOCK_M] running max of each row's scores (log2 units)
    l_i,            # [BLOCK_M] running sum of exp2(score - m_i) per row
    q,              # [BLOCK_M x D] this program's query rows (GQA packed, see the kernel below)
    q_pos,          # [BLOCK_M] absolute position of each row's query token
    start,          # position of the first key in this BLOCK_N slice
    k_cache_ptr,    # [NUM_PAGES x PAGE x Hkv x D]
    v_cache_ptr,    # [NUM_PAGES x PAGE x Hkv x D]
    bt_row_ptr,     # this request's row of the block table: logical page -> physical page
    kc_page_stride, # elements between pages     (PAGE * Hkv * D)
    kc_tok_stride,  # elements between tokens    (Hkv * D)
    kv_head_off,    # offset of our KV head inside one token's [Hkv x D] slot
    kv_hi,          # one past the last key this program may read
    scale_log2,     # softmax scale * log2(e)
    offs_n,         # [BLOCK_N] 0 .. BLOCK_N-1
    offs_d,         # [D]       0 .. D-1
    PAGE: tl.constexpr, # tokens per page
    MASKED: tl.constexpr # if we are doing diagonal or prediagonal
):
    """
    Core FlashAttention Algorithm wiuth Paging, whcih is just a gather from
    our kv_cache based on the block table indexes
    """

    ### absolute positions of the BLOCK_N keys in this slice ###
    kv_pos = start + offs_n

    ### PAGING. In our initial FlashAttn, a (batch, head) keys were one contiguous 
    ### [SEQ_LEN x HEAD_DIM] block. Here a requests keys live in a fixed sized pages
    ### scattered around the pool. 

    if MASKED:
        # Some KV positions may extend beyond kv_hi, where the block table has
        # no valid entries. Mask those loads and assign invalid positions page 0,
        # these placeholder keys are masked out later and never affect attention.
        #
        # Example: PAGE = 4, block_table = [1,7,2], kv_hi=10
        # this means so far this request is using 3 blocks that have 12 total positions
        # but only the first 10 of the 12 are valid (block 2 is not full yet)
        # The logical KV positions 0-3 map to page 1, positions 4-7 map to page 7
        # and positions 8-11 map to page 2
        # 
        # For kv_pos=[8,9,10,11] 
        #   kv_pos // PAGE = [2 2 2 2]
        #   kv_mask = [T T F F]
        #   page = [2 2 0 0]
        # positions 10 and 11 skip the block table load as they are out of range, 
        # we only have queries upto 10 so our kv cant go further than that.
        kv_mask = kv_pos < kv_hi
        page = tl.load(bt_row_ptr + kv_pos // PAGE, mask=kv_mask, other=0)
    else:
        # If we are not masked, these are prediagonal, in whcih case we have
        # already verified that everything exists. 
        #
        # Example: PAGE=4, block_table=[1, 7, 2].
        # For kv_pos=[4, 5, 6, 7]:
        #   kv_pos // PAGE = [1, 1, 1, 1]  # all index block_table[1]
        #   page           = [7, 7, 7, 7]  # physical page IDs
        # All positions are valid, so every block-table lookup is safe.
        page = tl.load(bt_row_ptr + kv_pos // PAGE)

    # Compute the starting element of each key vector in the paged KV cache. 
    # Page selects the physical page, kv_pos % PAGE selects the token's slot within
    # that page and kv_head_off selects the kv head's vector within the token
    # cast to int64 to avoid overflow on large kv pools
    kv_off = page.to(tl.int64) * kc_page_stride + (kv_pos % PAGE) * kc_tok_stride + kv_head_off

    ### Gather K and V, [BLOCK_N x D] each, one row per key
    if MASKED:
        k = tl.load(k_cache_ptr + kv_off[:, None] + offs_d[None, :], mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_cache_ptr + kv_off[:, None] + offs_d[None, :], mask=kv_mask[:, None], other=0.0)
    else:
        k = tl.load(k_cache_ptr + kv_off[:, None] + offs_d[None, :])
        v = tl.load(v_cache_ptr + kv_off[:, None] + offs_d[None, :])

    ### Scores is [BLOCK_M x BLOCK_N]. This is our QK.T, with the exp2 trick we used
    ### before so the softmax later can use exp2 rather than exp
    s = tl.dot(q, tl.trans(k)) * scale_log2

    if MASKED:
        ### Mask out any positions that are
        ###     1) invalid positions (kv_mask)
        ###     2) non causal after the diagonal
        s = tl.where((kv_pos[None, :] <= q_pos[:, None]) & kv_mask[None, :], s, float("-inf"))

    ### ONLINE SOFTMAX: the same recurrence as the original ###
    ### new running max: the larger of the old one and this slice's row max ###
    m_new = tl.maximum(m_i, tl.max(s, 1))

    ### this slice's probabilities, relative to the new max ###
    p = tl.math.exp2(s - m_new[:, None])

    ### how much to shrink what was accumulated under the old max: 2^(old - new) ###
    alpha = tl.math.exp2(m_i - m_new)

    ### rescale the old denominator, add this slice's row sums ###
    l_i = l_i * alpha + tl.sum(p, 1)

    ### rescale the old output, add P @ V. P goes to bf16 for the tensor cores (the ###
    ### original's P_block.to(fp16)), the accumulator stays fp32 ###
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

    return acc, m_new, l_i

@triton.jit
def _find_request(qsl_ptr, block, num_seqs, BLOCK_Q: tl.constexpr):
    """
    Binary search to find which request owns q_block block of the flattened prefill grid
    """
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(qsl_ptr + mid) // BLOCK_Q + mid <= block:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1

@triton.jit(do_not_specialize=["num_seqs"])
def _paged_attn_kernel(
    q_ptr,          # [T x H x D]  packed queries, all requests back to back
    k_cache_ptr,    # [NUM_PAGES x PAGE x Hkv x D]
    v_cache_ptr,    # [NUM_PAGES x PAGE x Hkv x D]
    out_ptr,        # [T x H x D]
    part_ptr,       # partial output for split-KV: [B x H x NUM_SPLITS x D], one row per request
    lse_ptr,        # logsumexp trick for split-KV: [B x H x NUM_SPLITS]
    qsl_ptr,        # [B+1] query_start_loc: request b owns rows qsl[b] : qsl[b+1] of q
    cu_k_ptr,       # [B+1] prefix sums of kv_len: request b's context is cu_k[b+1] - cu_k[b] long
    bt_ptr,         # block table: [B x max_pages_per_seq]
    num_seqs,       # B, including any padding requests (q_len 0) of a CUDA graph
    q_tok_stride,
    q_head_stride,
    o_tok_stride,
    o_head_stride,
    kc_page_stride,
    kc_tok_stride,
    kc_head_stride,
    bt_stride,
    p_tok_stride,
    p_head_stride,
    p_split_stride,
    l_tok_stride,
    l_head_stride,
    scale_log2,     # softmax scale * log2(e), for exp2
    PAGE: tl.constexpr,         # tokens per page
    GROUP: tl.constexpr,        # query heads per KV head (H // Hkv)
    D: tl.constexpr,            # head dim
    BLOCK_M: tl.constexpr,      # query ROWS per tile, a row is one (token, query head) pair
    BLOCK_N: tl.constexpr,      # keys per inner-loop slice
    NUM_SPLITS: tl.constexpr,   # split-KV: how many programs share one tile's context
    IS_DECODE: tl.constexpr,    # this launch takes the requests with q_len == 1, else q_len > 1
):
    """
    Launch up to two kernels to cover one batch, each handling its own kind of request:

        decode: GRID = (request, kv_head, split) -> q_len == 1
        prefill: GRID = (q_block, kv_head, 1) -> q_len > 1

    Both kernels recieve the same packed q and out tensors, but write to different query
    positions. Decode handles single token requests, while prefill handles multi-token
    requests.

    The grid depends on aggregate batch dimensions, not on how tokens are distributed
    across requests. This lets CUDA graphs reuse the same launch shape for different
    batches of the same padded size. Request boundaries and lengths live in GPU 
    metadata. Programs use this metadata to find their work and simply skip padding
    requests (q_len == 0) or out-of-range tiles


    The prefill grid flattens query blocks across requests. BLOCK_M indexes cover
    both query positions and query head groups: One unit of BLOCK_M covers
    GROUP query-head pairs, so the number of actual query timesteps per tile is:

        BLOCK_Q = BLOCK_M // GROUP

    For example, BLOCK_M = 8 and GROUP = 4, means 2 queries, each with 4 query heads
    per tile. BLOCK_Q is the number of query positions, not the number of query-head pairs

    Since each request starts its own sequence of query tiles (because one sequence doesn't
    attend to another), the grid will reserve T // BLOCK_Q + B candidate blocks. 

    Example: T = 100, B = 3, BLOCK_Q=16 with query lengths [40, 30, 30]. The packed request
    starts are then:

        qsl: [0, 40, 70, 100]
              A  B   C   END

    Each request is tiled independently:

        Request A (40 tokens)
            block 0: 16 tokens
            block 1: 16 tokens
            block 2: 8 tokens
        
        Request B (30 tokens)
            block 3: 16 tokens
            block 4: 14 tokens
            block 5: 0 tokens (skip)

        Request C (30 tokens)
            block 6: 16 tokens
            block 7: 14 tokens
            block 9: 0 tokens (skip)

    Why not use just qsl[b] // BLOCK_Q?

        Request A starts at token  0:  0 // 16 = 0
        Request B starts at token 40: 40 // 16 = 2
        Request C starts at token 70: 70 // 16 = 4

        This gives starting blocks [0, 2, 4]. But A already occupies
        blocks 0-2, so B would start at block 2 and overlap A's final
        block. C would likewise start before B's tiles have been fully
        assigned. Integer division maps packed token offsets to global
        16-token regions, but requests are tiled independently, and their
        boundaries need not align to multiples of BLOCK_Q.

    Instead, add the request index b:

        start_block[b] = qsl[b] // BLOCK_Q  + b

        A:  0 // 16 + 0 = 0
        B: 40 // 16 + 1 = 3
        C: 70 // 16 + 2 = 6

        The + b accounts for the preceding requests: each request starts
        a new sequence of tiles, so its block IDs must follow those already
        allocated to earlier requests. The resulting starts [0, 3, 6]
        give each request a separate, non-overlapping sequence of blocks.

    The grid launches T // BLOCK_Q + B = 100 // 16 + 3 = 9 programs, with IDs 0-8. 
    Block 5 and 8 are extra programs whose request relative tiles are entirely out
    of range so they would be skipped. Partial final tiles, such as request A Block 2, 
    process only the valid query positions


    Now we have to map the flattened block to its request and a local tile:

        start_block = [0, 3, 6]

        _find_request searches to find the last start_block that is <= the current
        flattened Block ID. Since the starts are sorted this identied the request
        owning that block:

        Example: Flattened Block 4

            start_blocks = [0, 3, 6]    
                            A  B  C

            A starts at 0 <= 4
            B starts at 3 <= 4
            C starts at 6 > 4

            Therefore block 4 belongs to request B which starts at 3. 

        We then conver the flattened block ID to a request relative tile

            tile = block - start_block[req] = 4-3 = 1

        Tile indices are zero based, so tile 1 is B's second tile, covering
        packed query positions [56,70), i.e. tokens 56-69.       

    Because the Grid Shape depends on aggregate dimensions (T, B, Hkv) while
    request boundaries and lengths are read from GPU metadata, allowing the same
    captured launch structure to handle different request distributions with the 
    same grid dimensions!       

    """

    ### Each query token contributes GROUP rows, one row per query head in its own group
    ### BLOCK_M counts these rows, not the query tokens, so we convert
    BLOCK_Q: tl.constexpr = BLOCK_M // GROUP

    ### Determine which request and which query tile this program processes.
    if IS_DECODE:
        ### During decode, we have one program per request, each non-padding
        ### request has exactly one query token so it only needs tile 0
        b = tl.program_id(0)
        tile = 0
    else:
        ### Prefill: program_id(0) is the flattened query-block ID shared across
        ### all requests, not a request index. We need to find which request
        ### owns this specific block. Each requests flattened starting block
        ### is qsl[b] // BLOCK_Q + b
        b = _find_request(qsl_ptr, tl.program_id(0), num_seqs, BLOCK_Q)

        ### Convert the flattened block ID to a tile index relative to 
        ### request_b by subtracting that request's starting block. For example
        ### start_blocks = [0, 3, 6], flattened block 4 belongs to request B (start_block 3)
        ### so the tile 4-3 = 1 means block 4 refers to tile index 1 of block B specifically
        tile = tl.program_id(0) - (tl.load(qsl_ptr + b) // BLOCK_Q + b)   # q-block within request b

    # The second grid dimension selects the KV head. Multiple query heads
    # may share this KV head when using grouped-query attention (GQA).
    kvh = tl.program_id(1) 

    ### The third grid dimension selects a slice ofthe KV context. with split-KV
    ### disabled, NUM_SPLITS = 1 and split is always 0
    split = tl.program_id(2) #


    ### Requests are packed back-to-back, without padding between requests. Unlike 
    ### standard fixed length FlashAttention, each request can have a different 
    ### number of query tokens and KV tokens. Prefix-sum arrays record where each
    ### request starts and ends in the packed tensors
    ###
    ### Example:
    ###   qsl     = [0, 3, 4, 8]
    ###   Request 0 queries: q[0:3] -> tokens 0, 1, 2 (3 queries)
    ###   Request 1 queries: q[3:4] -> token 3       (1 query)
    ###   Request 2 queries: q[4:8] -> tokens 4-7    (4 queries)
    ###
    ###   cu_k = [0, 10, 60, 67]
    ###   Request 0 KV: k/v[0:10]  -> 10 tokens
    ###   Request 1 KV: k/v[10:60] -> 50 tokens
    ###   Request 2 KV: k/v[60:67] -> 7 tokens
    ### 
    ### qsl_ptr and cu_k_ptr contain these prefix sums on the GPU, b
    ### identifies the request this program is responsible for!
    q_start = tl.load(qsl_ptr + b) ### What is the starting query for this specific request 
    q_len = tl.load(qsl_ptr + b + 1) - q_start ### What is the total number of queries in this request
    kv_len = tl.load(cu_k_ptr + b + 1) - tl.load(cu_k_ptr + b) ### What is the total number of KV in this request

    ### The decode and prefill launches handle disjoint request types. 
    ### Decode handles requests with exactly one token, Prefill has multiple tokens. 
    ### Padding requests (q_len=0) are skipped. The prefill grid is also an upper bound
    ### so some program can map to tiles beyond a requests actual query length and we exit
    ### without doing anything
    
    if IS_DECODE:
        ### If decoding q_len must be 1, otherwise it will run in the second pass or is invalid
        skip = q_len != 1
    else:
        ### If not decoding we skip all thing that are decode or invalid (1 or 0) and also
        ### this programs assigned tile is within the total number of queries for this request
        skip = (q_len < 2) | (tile * BLOCK_Q >= q_len)

    ### Exit otherwise
    if skip:
        return

    ### BLOCK_M indexes rows of (query token, query head) pairs.
    ### Each query token contributes GROUP rows, one for each query head
    ### sharing this KV head (grouped-query attention, GQA).
    r = tl.arange(0, BLOCK_M)

    ### Map each row to its query token, relative to this request.
    ### Example: GROUP = 4, BLOCK_Q = 2:
    ###   r:     [0, 1, 2, 3, 4, 5, 6, 7]
    ###   r//4:  [0, 0, 0, 0, 1, 1, 1, 1]
    ###
    ### If tile = 1, these rows represent query tokens [2, 3],
    ### with four query-head rows per token.
    q_idx = tile * BLOCK_Q + r // GROUP         # which of this request's query tokens [BLOCK_M]

    ### Map each row to its query-head index.
    ### Example: kvh = 0, GROUP = 4:
    ###   r%4:   [0, 1, 2, 3, 0, 1, 2, 3]
    ###   head:  [0, 1, 2, 3, 0, 1, 2, 3]
    ###
    ### If kvh = 2 and GROUP = 4, these would instead be heads [8, 9, 10, 11]
    ### repeated for each query token.
    head = kvh * GROUP + r % GROUP              # which query head                     [BLOCK_M]
    
    ### A tile can end partway through a query tokens group of rows, or the request
    ### can have fewer valid tokens than BLOCK_Q so we mask invalid rows
    ### r < BLOCK_Q * GROUP exclused unused rows if BLOCK_M is not divisible by GROUP
    ### q_idx < q_len exludes rows beyong this requests last token
    row_mask = (r < BLOCK_Q * GROUP) & (q_idx < q_len)   # [BLOCK_M]
    offs_d = tl.arange(0, D) # Feature dimensions of each query/key/value head [D]

    ### Load this tiles query vectors from the original packed q tensor. q_start
    ### maps request-relative query indices back to the packed token indices
    ### head selects the appropriate query head, offs_d selects the features
    q_off = (
        (q_start + q_idx).to(tl.int64)[:, None] * q_tok_stride
        + head[:, None] * q_head_stride
    )
    q = tl.load(
        q_ptr + q_off + offs_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )

    ### The query tensor contains only the new query tokens, but the KV cache
    ### contains both the earlier context and these new tokens.
    ###
    ### Therefore, the first new query is at absolute position kv_len - q_len.
    ### Adding q_idx gives the absolute position of each query in this request.
    ###
    ### Example: kv_len = 12, q_len = 4
    ###   New query positions: 8, 9, 10, 11
    ###   q_idx = [0, 1, 2, 3] -> q_pos = [8, 9, 10, 11]
    q_pos = kv_len - q_len + q_idx # [BLOCK_M]

    ### Find the last query token handled by this tile, clamped to
    ### the requests final valid query index
    ### Example: BLOCK_Q = 2, tile = 1, q_len = 3
    ###     this tile covers query indices [2, 3]
    ###     Only index 2 exists, so max_q = min(3, 2) = 2.
    max_q = tl.minimum((tile + 1) * BLOCK_Q - 1, q_len - 1)

    ### Compute teh exclusive KV boundary visible to the last query in this tile
    ### Causal attention lets a query at position P attend to keys 0...p, 
    ### so the exlusive endpoint is p+1
    kv_end = kv_len - q_len + max_q + 1 # [start: p+1] doesnt include right index (exlusive)

    ### Split the KV context across NUM_SPLITS programs to expose more parallelism
    ### when there are too few (request, head) programs to saturate the GPU.
    ### Round each split's chunk size up to a multiple of BLOCK_N. During prefill
    ### NUM_SPLITS = 1, so we are just rounding up to the nearest BLOCK_N
    chunk = tl.cdiv(tl.cdiv(kv_end, NUM_SPLITS), BLOCK_N) * BLOCK_N

    ### compute this splits starting position
    kv_lo = split * chunk

    ### compute the exclusive end position, clamped to kv_end so we never process beyond
    kv_hi = tl.minimum(kv_lo + chunk, kv_end)

    ### Running softmax state, one entry per row, all fp32 ###
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)     # running max
    l_i = tl.zeros([BLOCK_M], tl.float32)                   # running softmax denominator
    acc = tl.zeros([BLOCK_M, D], tl.float32)                # running output
    offs_n = tl.arange(0, BLOCK_N)
    bt_row_ptr = bt_ptr + b * bt_stride                     # this request's block table row
    kv_head_off = kvh * kc_head_stride                      # our KV head inside each token slot

    ### Use a two-pass strategy, as in our original FlashAttention kernel:
    ### process fully visible KV tiles first without causal masking, then process
    ### tiles that may overlap the causal diagonal or extend past the valid context
    ### with causal and bounds masks.
   
    ### Find the earliest query token handled by this tile, accounting for GROUP
    ### query-head rows per token.
    min_q = tile * BLOCK_Q

    ### Every query in the tile can attend to the prefix visible to its
    ### earliest query. A query at absolute position p sees keys [0, p],
    ### giving an exclusive boundary p + 1.
    ###
    ### Example: kv_len = 12, q_len = 4, tile = 1, BLOCK_Q = 2
    ###   The tile handles query indices 2 and 3 -> absolute positions 10, 11.
    ###   Its earliest query is at position 10, so every query in the tile
    ###   can see keys [0, 11): positions 0 through 10.
    ###   Thus all_visible = 11.
    all_visible = kv_len - q_len + min_q + 1

    ### Round all_visible DOWN to a BLOCK_N boundary.
    ### Complete KV tiles before this boundary are visible to every query
    ### in this query tile, so they can skip causal masking.
    ###
    ### Clamp the boundary to this split's [kv_lo, kv_hi] range.
    mid = tl.maximum(kv_lo, tl.minimum(kv_hi, (all_visible // BLOCK_N) * BLOCK_N))

    ### pass 1: [kv_lo, mid), every row sees every key, no masks at all ###
    for start in range(kv_lo, mid, BLOCK_N):
        acc, m_i, l_i = _attn_kv_tile(acc, m_i, l_i, q, q_pos, start,
                                      k_cache_ptr, v_cache_ptr, bt_row_ptr,
                                      kc_page_stride, kc_tok_stride, kv_head_off, kv_hi, scale_log2,
                                      offs_n, offs_d, PAGE, MASKED=False)
        
    ### pass 2: [mid, kv_hi), the diagonal and the ragged end, causal + bounds masks ###
    for start in range(mid, kv_hi, BLOCK_N):
        acc, m_i, l_i = _attn_kv_tile(acc, m_i, l_i, q, q_pos, start,
                                      k_cache_ptr, v_cache_ptr, bt_row_ptr,
                                      kc_page_stride, kc_tok_stride, kv_head_off, kv_hi, scale_log2,
                                      offs_n, offs_d, PAGE, MASKED=True)

    ### Normalise: divide by the softmax sum. Any empty rows here due to ragged batching
    ### remains 0, so we just replace 0 with 1 before dividing
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    ### If we had no splits (this is prefix or chunked prefix)
    if NUM_SPLITS == 1:
        ### the whole context was ours: this is the final output, write it ###
        o_off = (q_start + q_idx).to(tl.int64)[:, None] * o_tok_stride + head[:, None] * o_head_stride
        tl.store(out_ptr + o_off + offs_d[None, :], acc.to(out_ptr.dtype.element_ty),
                 mask=row_mask[:, None])
    else:
        # partial result + its log-sum-exp, an empty split reports -inf. Only decode
        # splits, so the request's one token is all there is: one buffer row per request
        req = b.to(tl.int64)
        p_off = req * p_tok_stride + head[:, None] * p_head_stride + split * p_split_stride
        tl.store(part_ptr + p_off + offs_d[None, :], acc, mask=row_mask[:, None])
        lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.math.log2(l_safe))
        tl.store(lse_ptr + req * l_tok_stride + head * l_head_stride + split, lse, mask=row_mask)

@triton.jit
def _combine_splits_kernel(
    part_ptr, # partial outputs for different splits [B x Hq x NUM_SPLITS x D]
    lse_ptr,    # LogSumExp of the Attention Score in Each Split [B x Hq x NUM_SPLITS]
    out_ptr,      # Where to store final combined output [T x Hq x D]
    qsl_ptr,      # [B+1] query_start_loc: where each request's token sits in out
    p_tok_stride,
    p_head_stride,
    p_split_stride,
    l_tok_stride,
    l_head_stride,
    o_tok_stride,
    o_head_stride,
    NUM_SPLITS: tl.constexpr,
    D: tl.constexpr
):
    """
    Grid = (request, query head)

    For one specific request and query head, we want to combine NUM_SPLIT vectors of
    length D into one vector of length D


    Each split s normalised its output over its own keys only:
        o_s = sum_{k in s} 2^(score_k) v_k / L_s      with  lse_s = log2(L_s)
    The true output normalises over all keys, so each split counts in proportion
    to its share of the total softmax mass:
        o = sum_s (L_s / sum_t L_t) * o_s = sum_s softmax_s(lse_s) * o_s
    """
    b = tl.program_id(0).to(tl.int64) # Which request?
    h = tl.program_id(1) # which head?
    s = tl.arange(0, NUM_SPLITS) # split dimension
    d = tl.arange(0, D) # embed dimension

    ### only the decode requests were split: skip prefill chunks and padding ###
    t = tl.load(qsl_ptr + b)                    # this request's (only) token in out
    if tl.load(qsl_ptr + b + 1) - t != 1:
        return

    ### load the logsumexp for every split ###
    # [NUM_SPLITS] -> [l0, l1, l2, l3]
    # where each l is the log-sum-exp of the attention scores for split s, because our attention kernel
    # uses base 2, we later use exp2 rather than exp
    lse = tl.load(lse_ptr + b * l_tok_stride + h * l_head_stride + s)

    ### softmax over splits, shifted by the max like any softmax. An empty split's ###
    ### -inf becomes weight exp2(-inf) = 0 ###
    w = tl.math.exp2(lse - tl.max(lse, 0)) 
    w = w / tl.sum(w, 0)

    ### the partial outputs [NUM_SPLITS x D], weighted and summed over the splits ###
    o = tl.load(part_ptr + b * p_tok_stride + h * p_head_stride + s[:, None] * p_split_stride + d[None, :])
    out = tl.sum(o * w[:, None], 0)
    tl.store(out_ptr + t.to(tl.int64) * o_tok_stride + h * o_head_stride + d, out.to(out_ptr.dtype.element_ty))

_paged_attn_prefill = triton.autotune(
    configs=_PREFILL_CONFIGS, key=["GROUP", "D", "PAGE"])(_paged_attn_kernel)
_paged_attn_decode = triton.autotune(
    configs=_DECODE_CONFIGS, key=["GROUP", "D", "PAGE", "NUM_SPLITS", "BLOCK_M"])(_paged_attn_kernel)

@functools.lru_cache
def _num_sms(device_index: int) -> int:
    ### streaming multiprocessors on this GPU (84 on an A6000), asked once and cached ###
    return torch.cuda.get_device_properties(device_index).multi_processor_count

def _num_splits(programs: int, max_seq_len: int, device_index: int) -> int:
    """
    We have to choose how many chunks to split the KV cache into for Split-KV in decoding.

    An SM (streaming multiprocessor) is a GPU execution unit containing many CUDA Cores and
    other hardware resources. CUDA cores execute the arithmetic instructions while SMs schedule
    and execute groups of threads in Warps. Triton launches the computations of 
    different pids and the GPU scheduler assigns these programs to available SMs.

    During decode, each program processee one request, one KV head and one KV context split.
    Without splitting, the grid has T * num_kv_heads programs which is too few to 
    keep the GPU busy when T is small (decoding one step for one request). So we aim for
    roughly 2 programs per SM, so the desired number of splits is ceil(2 * num_SMs / (batch_size * num_kv_heads)). 
    We cap the splits to avoid chunks smaller than 256 keys, as many smaller chunks will build a larger temporary
    and we make the largest number of splits be 16. 

    TLDR - split more, more parallel, more post attention reduction on temporaries
           split less, less parallel, less post attention reduction on temporaries 


    """

    want = triton.cdiv(2 * _num_sms(device_index), programs)
    splits = min(want, max(1, max_seq_len // 256), 16)
    return 1 if splits <= 1 else triton.next_power_of_2(splits)

def paged_attention(q, k_cache, v_cache, meta, scale, num_splits: int | None = None):
    """
    Drop-in for serve.py's attention backends
    
    q:          [T, H, D]                  packed queries for this step
    k/v_cache:  [num_pages, page, Hkv, D]  paged pool
    meta:       serve.AttnMetadata (query_start_loc, cu_seqlens_k, block_table,
                num_seqs, max_query_len, max_seq_len) 

    We do this with two launches, each taking its own kind of request (the kernel
    checks q_len on the GPU). The decode one takes requests with one new token (small
    tiles, split-KV). The prefill one takes the rest (which woudl have just been
    padding for 1 token requests). A Batch with only decodes can skip the second
    
    The launch shape comes from q's token count T, meta.num_seqs and meat.max_seq_len 
    only, never from how the tokens split into requests. For example if T = 16, 
    num_seqs = 4, and max_seq_len = 8192, we could have batch A which has 4
    requests of length [8 4 3 1], and a second batch B that has requests [2 2 6 6]. 

    If our launch grid was based on the longest request, that can change at any time, 
    and we would be limited to cuda graphs on decoding (when there is always 1 input)
    and then eager on prefills as we dont know the input length coming in

    You can read about this here as well in section 4.7: https://arxiv.org/pdf/2511.11581

        As the last optimization step, we modify the kernels to use
        a static launch grid, which consistently launches the same
        number of program instances. This was achieved by adapting 
        the number of Q blocks processed by each instance. The
        primary benefit of this approach is improved compatibility 
        with full CUDA or HIP graphs, as will be discussed in
        subsection 6.2
    
    """
    out = torch.empty_like(q)

    # first launch is only responsible for queries that are decoding
    _launch_attention(q, k_cache, v_cache, out, meta, scale, True, num_splits)
    if meta.max_query_len > 1:
        # second launch is only responsible for prefilling
        _launch_attention(q, k_cache, v_cache, out, meta, scale, False, None)
    return out

def _launch_attention(q, k_cache, v_cache, out, meta, scale, decode: bool, num_splits: int | None):
    """
    One launch over the whole batch. decode: the decode config with split-KV, taking
    the requests with one new token, else the prefill config, taking the others
    """
    ### shapes: T packed tokens, H query heads, page size, Hkv KV heads ###
    T, H, D = q.shape
    page, Hkv = k_cache.shape[1], k_cache.shape[2]
    group = H // Hkv
    B = meta.num_seqs
    assert group <= 64, "a prefill tile (BLOCK_M >= 64 rows) must hold at least one token's heads"

    ### When decoding, one Q tile holds a token's whole head group (q_len=1)
    ### and small batches split the context to keep every SM busy. tl.dot 
    ### needs atleast 16 rows on every dimension as a limitation so
    ### BLOKC_M = max(16, GROUP). For example, Qwen3-8B has 4 queries for each KV
    ### so for one timestep / one kv we load 4 queries. Therefore BLOCK_M = max(16, 4)
    ### this means 12 of teh rows are padding, but this is still better than loading KV 4
    ### times per each query head
    ### decode tile = 1 token x GROUP heads = GROUP real rows, padded up to BLOCK_M rows
    decode_block_m = max(16, triton.next_power_of_2(group))

    ### We can override the num_splits ###
    if num_splits is not None:
        splits = num_splits
    else:
        ### How many splits for KV cache if decoding, otherwise just 1
        splits = _num_splits(B * Hkv, meta.max_seq_len, q.device.index) if decode else 1

    ### outputs. In the original flash attention, we had to allocate a buffer for log-sum-exp
    ### as we needed to quickly recompute the softmax in the backward pass. We have no backward 
    ### pass, so if we are doing prefilling, no need for LSE at all. On the other hand, if 
    ### we are doing decoding, we need LSE again to store as an intermediate that is combined later
    ### (only decode requests are split, one token each, so the split buffers have one
    ### row per request, B rows, not T)
    if splits > 1:
        part = torch.empty(B, H, splits, D, dtype=torch.float32, device=q.device)
        lse = torch.empty(B, H, splits, dtype=torch.float32, device=q.device)
    else:
        part = lse = out  # unused

    ### the arguments both launches share (strides in elements, like the original) ###
    args = (
        q, 
        k_cache, 
        v_cache,
        out, 
        part, 
        lse,
        meta.query_start_loc, 
        meta.cu_seqlens_k, 
        meta.block_table,
        B,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        meta.block_table.stride(0),
        part.stride(0), part.stride(1), part.stride(2) if splits > 1 else 0,
        lse.stride(0), lse.stride(1),
        scale * 1.4426950408889634,  # log2(e)
    )

    if decode:
        ### one program per request (q_len = 1 fits in BLOCK_M), so the 3rd axis is ###
        ### just the splits. Prefill requests' programs exit right away ###
        _paged_attn_decode[(B, Hkv, splits)](
            *args,
            PAGE=page,
            GROUP=group,
            D=D,
            BLOCK_M=decode_block_m,
            NUM_SPLITS=splits,
            IS_DECODE=True,
        )
    else:
        ### flattened over q-blocks of BLOCK_Q = BLOCK_M // GROUP tokens: T // BLOCK_Q + B ###
        grid = lambda META: (T // (META["BLOCK_M"] // group) + B, Hkv, 1)
        _paged_attn_prefill[grid](
            *args,
            PAGE=page,
            GROUP=group,
            D=D,
            NUM_SPLITS=1,
            IS_DECODE=False,
        )

    if splits > 1:
        ### second kernel: merge each decode request's splits into out ###
        _combine_splits_kernel[(B, H)](
            part, lse, out, meta.query_start_loc,
            part.stride(0), part.stride(1), part.stride(2),
            lse.stride(0), lse.stride(1),
            out.stride(0), out.stride(1),
            NUM_SPLITS=splits, D=D, num_warps=4,
        )

@torch.inference_mode()
def autotune_prefill_attention(H: int, Hkv: int, D: int, page: int, dtype, num_tokens: int):
    """
    Run one synthetic prefill so the autotuner picks a prefill config now
    """
    n_pages = -(-num_tokens // page)
    dev = "cuda"
    k_cache = torch.randn(n_pages + 1, page, Hkv, D, device=dev, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(num_tokens, H, D, device=dev, dtype=dtype)
    lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
    meta = SimpleNamespace(
        query_start_loc=lens, cu_seqlens_k=lens,
        block_table=torch.arange(1, n_pages + 1, dtype=torch.int32, device=dev)[None],
        num_seqs=1, max_query_len=num_tokens, max_seq_len=num_tokens)
    paged_attention(q, k_cache, v_cache, meta, D ** -0.5)
    return _paged_attn_prefill.best_config

@torch.inference_mode()
def autotune_decode_attention(H: int, Hkv: int, D: int, page: int, dtype, max_seq_len: int):
    """
    Run one synthetic decode so the autotuner picks a prefill config now
    """
    ### 4 decoding requests with up to 1024 tokens of context each ###
    B, ctx = 4, min(max_seq_len, 1024)
    n_pages = -(-ctx // page)
    dev = "cuda"
    k_cache = torch.randn(B * n_pages + 1, page, Hkv, D, device=dev, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(B, H, D, device=dev, dtype=dtype)
    meta = SimpleNamespace(
        query_start_loc=torch.arange(B + 1, dtype=torch.int32, device=dev),
        cu_seqlens_k=torch.arange(0, (B + 1) * ctx, ctx, dtype=torch.int32, device=dev),
        block_table=(torch.arange(B * n_pages, dtype=torch.int32, device=dev) + 1).view(B, n_pages),
        num_seqs=B, max_query_len=1, max_seq_len=ctx)
    
    ### every power of 2 _num_splits can return, forced, so each NUM_SPLITS key is tuned ###
    for splits in (1, 2, 4, 8, 16):
        paged_attention(q, k_cache, v_cache, meta, D ** -0.5, num_splits=splits)

def moe_align_block_size(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """
    Group token-expert assignments by expert and pad each group to a multiple of 
    block_m, so every block_m - row tile belongs to one expert

    Example:

        topk_ids = [[1, 0], -> token 0 assigned to experts 1 and 0
                    [1, 2]] -> token 1 assigned to experts 1 and 2
        block_m = 4, num_experts = 3

        Flattened assignments: [1, 0, 1, 2]
                                0  1  2  3 <- assignment indices

        Expert 0: [1, PAD, PAD, PAD]
        Expert 1: [0, 2, PAD, PAD]
        Expert 2: [3, PAD, PAD, PAD]
        
        
        sorted_token_ids = [
            1, 4, 4, 4,   # Rows  0-3: expert 0
            0, 2, 4, 4,   # Rows  4-7: expert 1
            3, 4, 4, 4,   # Rows 8-11: expert 2
            4, 4, 4, 4,   # Rows 12-15: unused capacity
        ]

        expert_ids = [0, 1, 2, -1]
        num_post_padded = [12]

    """
    ### Flatten [T x top_k] into [T * topk]
    ### each element now represents one (token, topk slot) assignment
    ### its original token and slot can be recovered from its flat index
    ### token = index // topk, slot = index % topk
    flat = topk_ids.flatten()

    ### Map remote expert IDs (-1) to an extra bucket with ID num_experts. 
    ### In Expert Parallel, a token that needs an expert on a separate GPU
    ### will run on that gpu, we will all-reduce later. 
    flat = torch.where(flat < 0, num_experts, flat)

    ### There is one bucket per local expert + extra 1
    num_buckets = num_experts + 1

    ### Number of actual token expert assignments before we add padding
    numel = flat.numel()

    ### Each bucket needs most block_m - 1 padding grows to reach a multiple of block_m
    ### Reserve enough space for all assignments plus worst case padding for every 
    ### bucket, then round up to a multiple of block_m
    max_padded = triton.cdiv(numel + num_buckets * (block_m - 1), block_m) * block_m

    ### Count the number of assignments in each bucket. We use scatter_add_ as it increments
    ### counts[flat[i]] for every i assignment
    counts = torch.zeros(num_buckets, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))

    ### Round each buckets count up to a multiple of block_m, for example
    ### Example: counts=[1, 2, 1, 0], block_m=4 -> padded=[4, 4, 4, 0].
    padded = (counts + block_m - 1) // block_m * block_m

    ### Exclusive end position for each bucket in the padded output layout
    ### padded: [4,4,4,0] -> padded_end: [4,8,12,12]
    padded_end = padded.cumsum(0)

    ### Start position of each bucket in the padded layout
    ### [0 4 8 12]
    padded_start = padded_end - padded

    ### Start position of each bucket in the unpadded, sorted assignment list
    ### counts=[1,2,1,0] -> [0,1,3,4]. Bucket 0 ends at position 1, bucket 1 ends at position 3
    ### bucket 2 ends at position 4, bucket 3 ends in position 4
    count_start = counts.cumsum(0) - counts

    ### Sort the flattened assignment indices by their expert ID, the order contains the 
    ### original flat indices, not expert ids. stable=True preserves the order amont
    ### assignments for one expert. flat: [1, 0, 1, 2] -> [1, 0, 2, 3]
    order = torch.argsort(flat, stable=True)

    ### Expert ids of each assignment in sorted order
    ### [0, 1, 1, 2]
    experts_sorted = flat[order]

    ### arange gives each assignment's position in the sorted list.
    ### Subtract count_start for that assignment's expert to reset the rank
    ### to zero at the beginning of each expert's group.
    ###
    ### Example:
    ###   experts_sorted                  = [0, 1, 1, 2]
    ###   torch.arange(numel)             = [0, 1, 2, 3]
    ###   count_start[experts_sorted]     = [0, 1, 1, 3]
    ###   rank                            = [0, 0, 1, 0]
    rank = torch.arange(numel, device=flat.device) - count_start[experts_sorted]

    ###   padded_start[experts_sorted] = [0, 4, 4, 8]
    ###   rank                          = [0, 0, 1, 0]
    ###   dest                          = [0, 4, 5, 8]
    dest = padded_start[experts_sorted] + rank

    ### Allocate the padded output and fill every position with numel, Valid
    ### flat assignment indices are in [0, numel) so it can never be mistaken
    ### for a real assignment
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32, device=flat.device)

    ### Place each original flat ssignment index at its computed destination
    ### unwritten positions remain sentinel padding
    sorted_ids[dest] = order.to(torch.int32)

    ### Generate the starting row of each possible BLOCK_M tile.
    ### Example: max_padded=16, block_m=4 -> [0, 4, 8, 12].
    block_starts = torch.arange(0, max_padded, block_m, device=flat.device)

    ### Find the bucket containing each tile's starting row. right=True finds the first
    ### padded_end strictly greater than the row. A start exactly at an experts end therefore
    ### belongs to the next bucket
    expert_ids = torch.searchsorted(padded_end, block_starts, right=True)

    ### Any remote experts are changed to -1
    expert_ids = torch.where(expert_ids >= num_experts, -1, expert_ids).to(torch.int32)

    return sorted_ids, expert_ids, padded_end[-1:].to(torch.int32)

@triton.autotune(configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
                          for bn in (64, 128) for bk in (32, 64) for w in (4, 8) for s in (2, 3, 4)],
                 key=["N", "K", "BLOCK_M"])
@triton.jit
def _fused_moe_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    topk_weights_ptr,
    sorted_token_ids_ptr, 
    expert_ids_ptr, 
    num_post_padded_ptr,
    N, 
    K, 
    num_valid_tokens,
    stride_am, 
    stride_ak, 
    stride_be, 
    stride_bn, 
    stride_bk, 
    stride_cm, 
    stride_cn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, 
    TOP_K: tl.constexpr, 
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    """
    Tiled Matmul just like https://github.com/priyammaz/TritonKernels/blob/main/3_matmul.py

    A contains input_rows, B contains expert weights, and C recieves the output rows. 

        B: [E x N x K], with each expert E having weights [N x K]

    Each program computes a [BLOCK_M x BLOCK_N] output tile, reducing over K in chunks
    of BLOCK_K (the inner loop). Its M rows are assignments sorted by expert. 

    For the first GEMM, A has shape [T x K], each assignments flattened index
    is divided by TOP_K to recover its token row, since multiple expert assignments
    can share the same input token. 

    For the second GEMM, A has shape [T * TOP_K, K]. Here TOP_K=1 because each intermediate
    row already cooresponds to one assignment. 

    When MUL_ROUTED_WEIGHT is true, each output row is multiplied by its assignmen's router
    weight before being stored
    """

    ### One-dimensional launch: pid identifies this program in the full grid.
    pid = tl.program_id(0)

    ### Number of output feature tiles needed to cover N columns, the last tile
    ### may contain fewer than BLOCK_N valid columns
    num_pid_n = tl.cdiv(N, BLOCK_N)

    ### Number of assignment-row tiles in the launch grid, the grid contains
    ### num_pid_m * num_pid_n programs
    num_pid_m = tl.num_programs(0) // num_pid_n

    ### SAME TILIING LOGIC FOR TRITON MATMUL ###
    num_pid_in_group = GROUP_M * num_pid_n
    first_m = (pid // num_pid_in_group) * GROUP_M
    group_size = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_pid_in_group) % group_size
    pid_n = (pid % num_pid_in_group) // group_size

    ### sorted_token_ids is allocated for the maximum possible padded size,
    ### so the grid may include M tiles that this routing pattern does not need.
    ### num_post_padded is the actual number of padded rows in use.
    ### Example: allocated capacity = 16 rows, used rows = 12, BLOCK_M = 4.
    ### M tile 3 starts at row 12, so it exits without doing any computation.
    if pid_m * BLOCK_M >= tl.load(num_post_padded_ptr):
        return

    ### Global row indices for this programs BLOCK_M assignment rows. 
    ### pid_m = 2, BLOCK_M=4 -> rows [8 9 10 11]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    ### Load the original flattened assignment index for each row.
    ### Padding rows contain num_valid_tokens as a placeholder.
    ### Example: sorted_token_ids = [1, 4, 4, 4, 0, 2, 4, 4, ...],
    ###          offs_m = [4, 5, 6, 7] -> offs_token = [0, 2, 4, 4].
    offs_token = tl.load(sorted_token_ids_ptr + offs_m) # [BLOCK_M]

    ### Valid assignment indices are in [0, num_valid_tokens).
    ### The palceholder equals num_valid_tokens, so this mask excludes padding.
    ### Example: [0, 2, 4, 4] -> [True, True, False, False].
    token_mask = offs_token < num_valid_tokens

    ### Each M tile belongs to one expert because the assignments were grouped
    ### and padded by expert. Load that expert's ID from the alignment output.
    ### expert_ids[pid_m] = -1 marks the remote-assignment bucket or an unused
    ### tile, nonnegative values identify the local expert whose weights to use.
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    ### Output feature indices covered by this N tile.
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    ### Build Output pointers for each assignment row and output feature. 
    ### offs_token indexes C by flattened assignment, not by input-token row
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)

    ### Remote assignments are handled by the surrounding expert-parallel
    ### logic. This kernel writes zero for these rows and does no local GEMM.
    if expert < 0:
        tl.store(c_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty), mask=c_mask)
        return

    ### K indices for the current reduction tile.
    offs_k = tl.arange(0, BLOCK_K)

    ### Build pointers for the A and B tiles.
    ###
    ### A is indexed as [T, K] for the first GEMM or [T * TOP_K, K]
    ### for the second GEMM. We select BLOCK_M assignment rows and BLOCK_K
    ### features. Dividing by TOP_K maps assignment indices back to input-token
    ### rows in the first GEMM; TOP_K=1 makes this a no-op in the second.
    ###
    ### A tile shape: [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + (offs_token[:, None] // TOP_K) * stride_am + offs_k[None, :] * stride_ak

    ### B has shape [E, N, K]. Select this tile's expert, then address
    ### BLOCK_K reduction features and BLOCK_N output features.
    ### The pointers are arranged as [BLOCK_K, BLOCK_N] for tl.dot.
    b_ptrs = b_ptr + expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    ### Accumulate the output tile in FP32.
    ### A [BLOCK_M, K] tile multiplied by B [K, BLOCK_N] produces
    ### a [BLOCK_M, BLOCK_N] output tile. K is processed in BLOCK_K chunks,
    ### so the accumulator keeps the same shape across every iteration.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):

        ### Number of K features remaining, including this chunk.
        ### The last chunk may contain fewer than BLOCK_K valid features.
        k_rem = K - k * BLOCK_K
        
        ### Load A chunk: [BLOCK_M, BLOCK_K].
        ### Mask padding assignment rows and K positions beyond the end of K.
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)

        ### Load B chunk: [BLOCK_K, BLOCK_N].
        ### Mask K positions beyond the end of K and output columns beyond N.
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)

        ### Accumulate
        acc += tl.dot(a, b)

        ### Advance
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

def _moe_gemm(a, w, c, topk_weights, sorted_ids, expert_ids, num_post_padded,
              top_k: int, mul_weight: bool, block_m: int):

    ### B's shape is [E x N x K], N is the output width, K is the reduction width
    N, K = w.shape[1], w.shape[2]

    ### Launch one program for every possible M tile and every N tile.
    ### Use sorted_ids.numel() rather than the routing-dependent used length:
    ### this keeps the grid size independent of the actual routing decisions.
    ### Programs beyond num_post_padded exit early inside the kernel.
    grid = lambda META: (triton.cdiv(sorted_ids.numel(), block_m) * triton.cdiv(N, META["BLOCK_N"]),)

    ### Run Kernel
    _fused_moe_kernel[grid](
        a, w, c, topk_weights, sorted_ids, expert_ids, num_post_padded,
        N, K, topk_weights.numel(),
        a.stride(0), a.stride(1), w.stride(0), w.stride(1), w.stride(2),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, GROUP_M=8, TOP_K=top_k, MUL_ROUTED_WEIGHT=mul_weight,
    )

def fused_moe(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
              topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """
    x [T, H], w13 [E, 2I, H] (gate rows then up rows), w2 [E, H, I],
    topk_weights / topk_ids [T, top_k] (ids local to this GPU, -1 = another GPU's).
    Returns sum_k weight_k * expert_k(x) as [T, H], same as serve.moe_reference.
    """
    T, H = x.shape
    E, two_i, _ = w13.shape
    top_k = topk_ids.shape[1]

    ### decode: few rows per expert, so a tall tile would be mostly padding ###
    block_m = 16 if T * top_k <= 4 * E else 64
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(topk_ids, block_m, E)
    topk_weights = topk_weights.to(torch.float32).flatten()

    ### GEMM 1: every (token, slot) row through its expert's gate|up ###
    inter = torch.empty(T * top_k, two_i, dtype=x.dtype, device=x.device)
    _moe_gemm(x, w13, inter, topk_weights, sorted_ids, expert_ids, num_post_padded,
              top_k, False, block_m)

    act = silu_and_mul(inter[:, :two_i // 2], inter[:, two_i // 2:])

    ### GEMM 2: down projection, scaled by the router weight ###
    out = torch.empty(T * top_k, H, dtype=x.dtype, device=x.device)
    _moe_gemm(act, w2, out, topk_weights, sorted_ids, expert_ids, num_post_padded,
              1, True, block_m)

    ### each token's top_k rows are adjacent: sum them ###
    return out.view(T, top_k, H).sum(1)

@torch.inference_mode()
def autotune_fused_moe(w13: torch.Tensor, w2: torch.Tensor, top_k: int, max_num_tokens: int):
    """
    Tune both GEMMs for both tile heights (BLOCK_M 16 for decode, 64 for prefill)
    with random routing on the real expert weights. The profiling forward routes
    every token to the same few experts (its inputs are zeros), which would tune
    for the wrong problem, and on an expert-parallel GPU would hit no local expert.
    """
    E, H = w13.shape[0], w13.shape[2]
    for T in (1, max_num_tokens):
        x = torch.randn(T, H, device=w13.device, dtype=w13.dtype)
        ids = torch.randint(0, E, (T, top_k), device=w13.device)
        weights = torch.rand(T, top_k, device=w13.device)
        fused_moe(x, w13, w2, weights, ids)

@triton.jit(do_not_specialize=["seed"])
def _sample_topk_kernel(
    vals_ptr, 
    idx_ptr, 
    out_ptr, 
    temp_ptr, 
    topk_ptr, 
    topp_ptr,
    vals_stride, 
    idx_stride, 
    temp_stride, 
    topk_stride, 
    topp_stride,
    K, 
    seed, 
    BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)

    ### this row's K candidates, already sorted best first by torch.topk ###
    v = tl.load(vals_ptr + row * vals_stride + offs, mask=offs < K, other=float("-inf")).to(tl.float32)

    ### this row's settings. Strided: in serve.sample they are columns of one [B, 4]
    ### tensor (settings.unbind(1)), so row r's top_k is NOT at topk_ptr + r ###
    t = tl.load(temp_ptr + row * temp_stride)
    k = tl.load(topk_ptr + row * topk_stride)
    p = tl.load(topp_ptr + row * topp_stride)

    ### temperature: higher -> flatter ###
    v = v / t

    ### this row's own top_k (the batch's K is the largest of them) ###
    v = tl.where(offs < k, v, float("-inf"))

    ### top-p on what top-k kept: drop a candidate once the ones before it reach p ###
    probs = tl.exp(v - tl.max(v, 0))
    probs = probs / tl.sum(probs, 0)
    before = tl.cumsum(probs, 0) - probs
    v = tl.where(before > p, float("-inf"), v)

    ### Gumbel-max: argmax(v + Gumbel noise) is a draw from softmax(v) ###
    u = tl.rand(seed, (row * BLOCK + offs).to(tl.int32))           # uniform, a different stream per element
    u = tl.minimum(tl.maximum(u, 1e-10), 1.0 - 1e-7)
    choice = tl.argmax(v - tl.log(-tl.log(u)), 0)

    tl.store(out_ptr + row, tl.load(idx_ptr + row * idx_stride + choice))

def sample_from_topk(vals: torch.Tensor, idx: torch.Tensor, temperature: torch.Tensor,
                     top_k: torch.Tensor, top_p: torch.Tensor):

    B, K = vals.shape
    out = torch.empty(B, dtype=torch.int64, device=vals.device)
    seed = int(torch.randint(0, 2**31 - 1, (1,)))      # host RNG: honours torch.manual_seed
    _sample_topk_kernel[(B,)](vals, idx, out, temperature, top_k, top_p,
                              vals.stride(0), idx.stride(0),
                              temperature.stride(0), top_k.stride(0), top_p.stride(0), K, seed,
                              BLOCK=max(16, triton.next_power_of_2(K)), num_warps=1)
    return out
