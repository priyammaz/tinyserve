"""
tinycomm: tinyserve's own all-reduce for tensor parallelism, for GPUs that can 
read and write each other's memory (NVLink or PCIe peer-to-peer). This may be a new
idea for anyone that hasn't seen this so lets break down the motivation here:

What is this?

   Tensor Parallelism requires an All-Reduce on the out_proj of attention or on
   the down_proj of either the MLP or MoE. This means the GPU's need to communicate
   sharing their portion of that accumulation to then add together. Every layer
   in our Transformer has 2 of these reductions then, so on Qwen3-8B with 36 layers
   there are 72 separate all-reduce calls! This gets expensive very quickly. 

Why not use NCCL?

    NCCL is Nvidia's own collective library, and anytime we do DDP/FSDP or whatever
    we are using NCCL. The challenge is that NCCL is highly optimized for LARGE data
    shares (think training models with large tensors, large gradients, etc...) not really
    for small data (like 1 token decodes). NCCL has its own fixed cost that you can 
    actually measure here: https://github.com/NVIDIA/nccl-tests

    On my system I measure that moving 8kb of data via NCCL takes 8.8 microseconds. When
    doing the transfer through torch (which wraps NCCL) it takes 14.7 microseconds. At the
    rated 112 Gbps of the NVLink bridge connecting my two GPUs, this should in the best
    case scenario take only 0.15 microseconds, so we have something to improve here. 

    This overhead is for obvious reasons, NCCL is a general communication library, that 
    supports all sorts of interconnects (NVLink, PCIe, Ininiband, ...), any number of 
    GPUs or Nodes, every data type and operation. But we have only one operation in mid
    all-reduce... so lets build a reduction collective for that alone!

Peer-to-Peer Memory

    With NVLink (pr PCIe P2P) a kernel running on GPU0 can load from and store to GPU1's 
    memory with ordinary loads and stores as long as we have the pointers to those
    memory locations in other GPUs. This is build through PyTorch and their Symmetric 
    Memory: https://docs.pytorch.org/docs/2.15/symmetric_memory.html

        t = symm_mem.empty(n, ...)               every GPU allocates a buffer of the same size
        h = symm_mem.rendezvous(t, group)        the GPUs swap handles; each maps the others' buffers
        h.buffer_ptrs                            [address of rank 0's buffer, address of rank 1's]


How Do We Know When the Other GPUs Data Has Arrived?

    Transfering data between GPUs is easy, but synchronizing is much more challenging. GPU1 
    cant just read GPU0's buffer, it could be GPU0 isn't done writing yet, or the buffer
    still holds the values from a previous forward pass. Each GPU needs a way to say that
    the data for this specific call is there, and for this we will build 2 ways:


    1. _ptp_all_reduce_kernel

        Write my data into a buffer and raise a flag in each peer's memory saying that
        call c is ready. Wait for every peers flag, and then read every buffer and accumulate
        them up. This means we have plain data, no wasted bytes, but there are more round trips

    2. _p2p_ll_all_reduce_kernel

        The flag for the data travels along size it. Every 4 bytes of data has another 4 byte flag
        attached to it in a single 8 byte sotre, when the reader sees the right call number, that
        means the data has arrived. This means only one trip across NVLink, but we double the bytes
    
    _ptp_all_reduce_kernel is ideal for our larger cases (things like prefilling) as we dont want
    to double the memory transfer and the larger compute cost of the operations better hide 
    communication latency. But when doing just decoding, computation is fast, we are better of
    doing only a single pass of more data to reduce the overall latency. Specifically if the 
    transfer is less than 64K we use the fast path

        
Some definitions to make things easier:

    rank: Which GPU this process runs on
    WORLD: How many ranks there are
    peer: any other rank
    call c: how many times this kernel has run, so every rank makes the same all reduce calls
            in identical order (so call 7 on rank 0 and call 7 on rank 1 belong to the same allreduce)
    word: not a token, just a comp arch term that means a fixed-sized datum
"""

import time

import torch
import triton

try:
    # only available after torch 2.9
    import torch.distributed._symmetric_memory as symm_mem
except ImportError:
    symm_mem = None

import triton.language as tl

@triton.jit
def _unpack(d, DT: tl.constexpr):
    """
    One int32 word (4 bytes) has 2 2-byte values packed in it. This splits them apart and
    returns each one as fp32. The low half is element 2o, the high half is element 2o+1


    A float value is stored with its bits split into three fields:

        sign: 1 bit (+ or -) where 0 means positive (-1)^0 = 1 and 1 means negative (-1)^1
        exponent: 8 bits that controls the scale of the number by multiplying/dividing by powers of 2
                  With 8 bits here we can have a total range from -127 to 127. In actuality we store 
                  0 - 255 and use a default -127 bias (IEEE standard) but thats not super important
                  to what we are doing here. 
        Mantissa: Remaining 23 bits. Right now our exponent gives us the scale (magnitudes of 2)
                  but the mantissa provides fine-grained precision within that scale.

    Now the layout for the types we care about are:

        format   sign   scale    detail   total
        fp32     1      8 bits   23       32 bits
        bf16     1      8 bits    7       16 bits
        fp16     1      5 bits   10       16 bits

    Notice that bf16 is just fp32 with the detail cut short? Same sign, same 8bit scale, 
    but only the first 7 detail bits. So two numbers in bf16 vs fp32 are the same except for
    the finer tail end precision. This is why bf16 works so well for neural networks, we get
    the same range of values as fp32, but not the inner detail, which models seem robust to. 

    Putting the bf16 bits on top with zeros below is the fp32 number. 

        1.5 in fp32:   0 01111111 1000000 0000000000000000    0x3FC0_0000
        1.5 in bf16:   0 01111111 1000000                     0x3FC0       <- the first 16 bits

    On the other hand, fp16 needs to convert. its fields dont line up with fp32 and moving bits
    would give different numbers. 

    """
    
    if DT == tl.bfloat16:
        # d << 16 takes the lower 16 bits and move them up, the upper 16 
        # bits dissapear and the lower 16 bits are now 0. So we basically
        # moved the lower to upper and then read it again
        lo = (d << 16).to(tl.float32, bitcast=True)
        
        # d & -65536 (~0xFFFF) is bit masking, -65536 is [1 1 1...][0 0 0 ...]
        # where the first 16 bits are 1 and the last 16 bits are all 0
        # this way when we do bitwise and (&) we keep the first 16 bits
        # and zero out the last 16 bits. This is our upper 16 bits then
        hi = (d & ~0xFFFF).to(tl.float32, bitcast=True)
    else:
        # .to(tl.int16) only keeps the lower 16 bits of d, which is an int32. Normally
        # that would lose information, but the bits we remove are just the other value
        # in the word. Then we reinterpret those 16 bits as the float16 that they
        # were, and cast to float32
        lo = d.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)

        # same idea, but start by grabbing the 16 bits we want, by shifting
        # the entire 32 bit word right by 16 positions
        hi = (d >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    return lo, hi

@triton.jit
def _pack(lo, hi, DT: tl.constexpr):
    """
    two fp32 tensors -> one int32 word per pair
    
        .to(tl.bfloat16) -> convert float32 accum back to bfloat16
        .to(tl.uint16, bitcast=True) -> bitcast to uint16 (keeps binary identical)
        .to(tl.int32) -> promote 16 bit to 32 bit, filling the upper with zeros
    """
    lo_bits = lo.to(DT).to(tl.uint16, bitcast=True).to(tl.int32)
    hi_bits = hi.to(DT).to(tl.uint16, bitcast=True).to(tl.int32)

    # both hi and lo populate the lower 16 bits, with 0 in the upper, move
    # the hi up and then all 32 are occupied
    return lo_bits | (hi_bits << 16)

@triton.jit
def _own_values(x_ptr, o, mask, row, TOPK: tl.constexpr):
    """
    Element o of this rank's [T,row] partial sums. If TOPK > 1 then we have a second
    reduction over the TOPK, which we now fuse here instead of the separate .sum()
    run in that kernel. 

    o is a vector of flat output indices (BLOCK of them), mask says which are inside 
    the actual message we are processing
    """
    # Dense layers are already [T, row], elelent o is just x[o]
    if TOPK == 1:
        return tl.load(x_ptr + o, mask=mask, other=0.0)

    # Otherwise we have an accumulate first. We need to map o to the cooresponding
    # expert rows. if row = 2048, o = 5, we have t = 0, h = 5 and the 8 expert
    # rows of token 0 are rows 0...7 of x, so we want x[0, 5] + ... + x[7, 5]
    t = o // row
    h = o % row

    # load the starting point (expert 0 index)
    acc = tl.load(x_ptr + t * TOPK * row + h, mask=mask, other=0.0).to(tl.float32)

    # loop through the rest
    for k in tl.static_range(1, TOPK): # static_range: unrolled at compile time as TOPK is known
        acc += tl.load(x_ptr + (t * TOPK + k) * row + h, mask=mask, other=0.0).to(tl.float32)

    return acc.to(x_ptr.dtype.element_ty)

@triton.jit
def _own_words(x32, o, mask, row_words, TOPK: tl.constexpr, DT: tl.constexpr):
    """

    Return this rank's contribution for packed word o

    Each int32 word stores two adjacent bf16 values:
        bits [15:0] -> lower address value (element 2o)
        bits [31:16] -> next value (element 2o + 1)
    
    Example:
        element 0 = 1.5 -> bf16 bits 0x3FC0
        element 1 = 1.0 -> bf16 bits 0x3F80

        packed = 0x3F80_3FC0

    Lets do an example for when TOPK > 1 as its more complicated:

        x: [T, TOPK, E] where each element is in bf16

        Let T = 2, TOPK = 2, E = 4 -> x: [2 x 2 x 4]

        Token 0: 
            Expert 0: [a0 a1 a2 a3] <- not literally expert 0, but the first of its topk
            Expert 1: [b0 b1 b2 b3] <- not literally expert 1, but the second of its topk
        
        Token 1:
            Expert 0: [c0 c1 c2 c3] 
            Expert 1: [d0 d1 d2 d3]

        The communication kernel we will write works with  32-bit words, each word stores
        two adjacent bf16 values along E:

            word 0: [element 0, element 1]
            word 1: [element 2, element 3]
        
        Therefore, the packed input has logical shape:
            
            [T x TOPK, E // 2] int32

        For our example:

        Token 0:
            Expert 0: [pack(a0, a1), pack(a2, a3)]
            Expert 1: [pack(b0, b1), pack(b2, b3)]

        Token 1:
            Expert 0: [pack(c0, c1), pack(c2, c3)]
            Expert 1: [pack(d0, d1), pack(d2, d3)]

        We call the pack(a0, a1) a "word" and we will have
        in this case 2 words per row (which we call row_words)

    
    TOPK = 1:
        The input is already one packed word per output position
        so we can return it directly

    TOPK > 1:
        The input contains TOPK expert rows PER token, unpack each
        experts word into two fp32 values, sum the lower and upper
        bf16 values independently across the experts, then repack
        it back the two sums into one int32 word

        For example we need to do a0 + b0, a1 + b1, but they are currently
        packed together. So we simply unpack

            a0, a1 = unpack(pack(a0, a1))
            b0, b1 = unpack(pack(b0, b1))

            o0 = a0 + b0
            o1 = a1 + b1

            return pack(o0, o1)

    """

    # With TOPK = 1, there is no expert-row reduction to perform, the
    # original data x: [T, E] in bf16, but because of packing
    # it is actually [T, E//2] in int32, where each int32 is two packed bf16
    # values. There is no expert reduction to perform so we just return it as is!
    if TOPK == 1:
        return tl.load(x32 + o, mask=mask, other=0)

    # For TOPK > 1, we first sum the expert contributions for each 
    # token before communicating the results
    #  
    # Original Data 
    #   x: [T, TOPK, E] in BF16
    # We pack every 2 adjacent features along E into a single int32
    #   x32: [T x TOPK, E // 2]
    # We can then define row_words = E // 2
    # Each token has TOPK expert rows, with row_words packed
    # words per row, the input is flattened in token-major order
    #
    #   token 0, expert 0: row_words # of words
    #   token 0, expert 1: row_words # of words
    #   token 1, expert 0: row_words # of words
    #   token 1, expert 1: row_words # of words
    #   ...
    #
    # however, o indexes the output after the reduction happens, the output
    # here being [T x E//2] packed int32 words
    # so we just need to recover the token index t and packed word positions h
    # within that tokens feature dimension
    # Example: 
    #   o = 0 -> token 0, word 0 (features 0 and 1)
    #   o = 1 -> token 0, word 1 (features 2 and 3)
    #   o = 2 -> token 1, word 0 (features 0 and 1)
    #   o = 3 -> token 1, word 1 (features 2 and 3)
    t = o // row_words # which token?
    h = o % row_words # which word?

    # create two fp32 accumulators. Each packed word contains two bf16 values
    #   lower 16 bits -> feature 2*h
    #   upper 16 bits -> features 2*h + 1
    # lo accumulates fetures 2*h across experts, and hi accumulates features 2*h+1
    # across the experts. The accumulators have the same shape as o so every triton
    # lane will handle its own output word positions
    lo = tl.zeros(o.shape, dtype=tl.float32)
    hi = tl.zeros(o.shape, dtype=tl.float32)

    # Iterate over the expert contributions for this token. For a given token t and word
    # position h, load the same word position from each of the TOPK expert rows
    for k in tl.static_range(TOPK):

        # x32: [T x TOPK x E // 2] 
        # Token 0:
        #     Expert 0: [pack(a0, a1), pack(a2, a3)]
        #     Expert 1: [pack(b0, b1), pack(b2, b3)]

        # Token 1:
        #     Expert 0: [pack(c0, c1), pack(c2, c3)]
        #     Expert 1: [pack(d0, d1), pack(d2, d3)]

        # if t = 1, k = 0,  and we know row_words = 2, TOPK = 2
        # (t * TOPK + k) = 1 * 2 + 0 = 2 which tells us we want
        # to have Token 1 expert 0 here. But, each expert has 2 (row_words)
        # elements in it, so to go from one expert to the next we need to move
        # over by 2 elements at a time (stride basically). So (t * TOPK + k) * row_words
        # moved me over to the starting position of expert 0 in token 1. k controls which
        # expert in token 1 I want (the inner loop), h controls which packed chunk (word)
        # do we want
        d = tl.load(x32 + (t * TOPK + k) * row_words + h, mask=mask, other=0)

        # unpack and add
        d_lo, d_hi = _unpack(d, DT)        
        lo += d_lo
        hi += d_hi

    # repack the accumulation
    return _pack(lo, hi, DT)
        
@triton.jit
def _next_call(count):
    """
    Return the next synchronization call number, wrapping from 2**30 back to 1

    The 30 bit keeps only the lowest 30 bits of counting, adding 1 ensures 
    the result is always in [1, 2**30], never 0

        count          next call
        0              1
        6              7
        2**30 - 1      2**30
        2**30          1       <- wraparound

    Why use a 30bit mask?

        The counter is stored in int32, whose max positive value is 2 ** 31 - 1. The 
        mask is applied before adding 1, so a 31 bit mask could produce 2 ** 31 - 1, 
        and if we add 1 to that we overflow!

    We have a few requirements for the counter:

        1. Never zero, zero flags and buffers represent "not yet written"
        2. always positive, as we will do modulo

    """
    return (count & 0x3FFFFFFF) + 1

@triton.jit
def _p2p_all_reduce_kernel(
    x_ptr,       # [n] this rank's partial sums ([n / row, TOPK, row] with TOPK > 1)
    out_ptr,     # [n] the total (x itself unless TOPK > 1)
    data_ptrs,   # [WORLD] int64: address of each rank's shared data buffer, as mapped on this GPU
    flag_ptrs,   # [WORLD] int64: address of each rank's flags, as mapped on this GPU
    counter_ptr, # [max blocks] int32: how many calls each block has made
    n,
    half,        # elements in one half of a data buffer
    row,         # the hidden size (only used with TOPK > 1)
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr = 1,        # MoE: also sum each token's top_k expert rows
    SPIN_LIMIT: tl.constexpr = 20_000_000,   # How long to poll, then report and give up
):

    # Which part of the message belongs to this block. Imagine we flatten X into one
    # long vector, every block owns one contiguous slice of it and walks through 
    # its slice one BLOCK value at a time. Lets say a flattened x has 200 elements, 
    # and we launch with a grid of 0-3 (4 blocks) each processing its assignment 
    # 16 at a time
    pid = tl.program_id(0)                          # which block of x are we processing? lets say 1
    chunks = tl.cdiv(n, BLOCK)                      # how many chunks in total? ceil(200/16) = 13
    per = tl.cdiv(chunks, tl.num_programs(0))       # how many chunks does each pid process? ceil(13 / 4) = 4
    lo = pid * per * BLOCK                          # where my slice starts (1 x 4 x 16 = 64)
    hi = tl.minimum(lo + per * BLOCK, n)            # min(64 + 4*16, 200) = 128
    offs = tl.arange(0, BLOCK)                      # offset for each chunk of BLOCK
    dtype = x_ptr.dtype.element_ty

    # This blocks call number
    c = _next_call(tl.load(counter_ptr + pid))

    # before we write this number, we have to ensure everyone involved has read it first, otherwise a late
    # warp will have threads that read the new c+1 instead of c
    tl.debug_barrier()
    tl.store(counter_ptr + pid, c)

    # Alternate between two halves of our buffer
    base = (c % 2) * half # offset over, either start at 0 or half

    # copy my chunks of x into the correct half of my shared buffer where peers can later read them
    # this is a fast write going to my local memory
    # We are loading a thing that contains literally a Pointer. We offset over to RANK (this gpus rank)
    # and then offset over to the half we will write to
    mine = tl.load(data_ptrs + RANK).to(tl.pointer_type(dtype)) + base
    for start in range(lo, hi, BLOCK):
        o = start + offs
        tl.store(mine + o, _own_values(x_ptr, o, o<n, row, TOPK), mask=o<n)

    # Cant keep going until all the threads are done writing.
    tl.debug_barrier()

    # raise my flag in every peer's memory telling it that my block pid has finished writing call c
    # 
    # every GPU has its own copy of the flags table, its a 2D table of [block, sender], flattened
    # into one long list. Each entry is its position in that list. For example:
    #
    #   flags[block, sender]      sender 0   sender 1
    #   block 0                      0          1
    #   block 1                      2          3
    #   block 2                      4          5
    #   ...
    #   block 5                     10         11
    #
    # flags[block, sender] holds the last call that sender's block finished writing. To find my entry
    # each row is WORLD entries wide, so pid * WORLD skips to my blocks row (pid's row) and + RANK
    # moves over to my column.
    # The problem is a plain store isnt enough. The code basically is, copy the data and then raise
    # a flag, but while a large chunk of data is being copied, the flag can overtake it and be sent
    # to the peers which would be wrong, flag should only go AFTER ALL the copies are done
    #   atomic_xchg:    swaps a new value into memory and hands back the old one which we ignore. 
    #                   A plain tl.store() would work but it cant promise when the Peer sees them compared
    #                   to our other writes. Atomic Exchange can through sem and scope.
    #   sem="release"   Ensure all prior memory reads/writes are completed BEFORE this operation takes place
    #   scope="sys"     Sync across the system, writes are visible system-wide
    # sys is expensive, it has to wait until our earlier writes are visible to the whole system.
    # The flag-in-data kernel doesn't pay for this: its flag and data land together in one 8 byte
    # store, so there's nothing to keep in order
    for r in tl.static_range(WORLD):
        if r != RANK: # no need to tell myself
            peer = tl.load(flag_ptrs + r).to(tl.pointer_type(tl.int32))  # r's table, as mapped on this GPU
            tl.atomic_xchg(peer + pid * WORLD + RANK, c, sem="release", scope="sys")
        
    # Now we wait until every peer has raised its flag for this block in MY memory! Polling my own memory
    # is cheap, we are waiting on the peers to write to me to tell me they are done on their end. We accept
    # either c (the current call) or nxt (c+1) as the peer could be one call ahead (rank 1's block 6 can
    # see my flag 3, finish call 3, start call 4 and write that before my block ever looks. This is why
    # we double buffer)
    nxt = _next_call(c)

    # my own flags table, the ones my peers write into
    me = tl.load(flag_ptrs + RANK).to(tl.pointer_type(tl.int32))

    for r in tl.static_range(WORLD):

        # I never raise a flag fro myself, so waiting on my own column would wait forever
        if r != RANK:

            # has peer r finished this call for my block? flags[pid, r] in my own table
            # hodsl the last call r's block pid finished and r writes it there itself. 
            # we add 0 so its just a return, same as tl.load() but we can add sem=acquire
            # which ensures no future memory reads occur before reasing that same flag. 
            # unlike "release" from earlier which is prior
            f = tl.atomic_add(me + pid * WORLD + r, 0, sem="acquire", scope="sys")

            # how many times we have looked
            spins = 0

            # keep going while the flag is neither c nor nxt
            while (f != c) & (f != nxt) & (spins < SPIN_LIMIT):
                # look again. This is the same read as above, it has to be atomic + acquire every
                # time, as the read that finally sees the flag is the one that must give us the data
                f = tl.atomic_add(me + pid * WORLD + r, 0, sem="acquire", scope="sys")
                spins += 1

            # we left the loop but never got what we needed
            if (f != c) & (f != nxt) & (pid < 2):
                tl.device_print("P2P ALL-REDUCE STUCK rank/block/call/peer_flag", RANK, pid, c, f)

    # polling above is done
    tl.debug_barrier()

    # read the same chunks from every rank and add them up (the peers over NVLINK)
    for start in range(lo, hi, BLOCK):
        o = start + offs
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for r in tl.static_range(WORLD):
            src = tl.load(data_ptrs + r).to(tl.pointer_type(dtype)) + base
            acc += tl.load(src + o, mask=o < n, other=0.0).to(tl.float32)
        tl.store(out_ptr + o, acc.to(dtype), mask=o < n)

@triton.jit
def _p2p_ll_all_reduce_kernel(
    x_ptr,        # [n] bf16 or fp16 partial sums ([n / row, TOPK, row] with TOPK > 1)
    out_ptr,      # [n] the total (x itself unless TOPK > 1)
    ll_ptrs,      # [WORLD] int64: each rank's receive buffer, as mapped on this GPU
    counter_ptr,  # [max blocks] int32: how many calls each block has made
    n_words,      # n / 2: the message as 32-bit words, two values each
    slot_words,   # words in one half of one sender's slot in a receive buffer
    row_words,    # the hidden size / 2 (only used with TOPK > 1)
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,   # words per step (not values!)
    TOPK: tl.constexpr = 1,
    SPIN_LIMIT: tl.constexpr = 20_000_000,
):

    # same setup as before (but on packed n_words rather than just n)
    pid = tl.program_id(0)                          
    chunks = tl.cdiv(n_words, BLOCK)                
    per = tl.cdiv(chunks, tl.num_programs(0))       
    lo = pid * per * BLOCK                          
    hi = tl.minimum(lo + per * BLOCK, n_words)            
    offs = tl.arange(0, BLOCK)                      

    # same counter as before
    c = _next_call(tl.load(counter_ptr + pid))
    tl.debug_barrier()
    tl.store(counter_ptr + pid, c)

    # the flag is packed into the top 32 bits of each 8 byte word
    flag = c.to(tl.int64) << 32
    half = (c % 2) * slot_words

    # View the bf16 (of fp16) tensors as int32
    DT: tl.constexpr = x_ptr.dtype.element_ty
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    out32 = out_ptr.to(tl.pointer_type(tl.int32))

    # Peers will write into here, 32 bits of flag, 32 bits of data
    me = tl.load(ll_ptrs + RANK).to(tl.pointer_type(tl.int64))

    # my slice one chunk at a time
    for start in range(lo, hi, BLOCK):

        # the word indices of this chunk
        o = start + offs

        # build the 8 byte words we push to the peer. 4 bytes of call number on top and 
        # 4 bytes of data (two bf16/fp16 values) on the bottom
        #   _own_words() is my 4 bytes of data as an int32
        #   .to(tl.int64) widens it to 8 bytes, but it does this by just copying the sign bit
        #                 repeatedly, if positive it padds with zeros, if negative then 1
        #   & 0xFFFFFFFF: keep the bottom 4 bytes and zero the top 4, so the top half is
        #                 empty no matter what the cast filled it with 
        word = flag | (_own_words(x32, o, o < n_words, row_words, TOPK, DT).to(tl.int64) & 0xFFFFFFFF)

        # send chunk to every peer (again unrolled at compile time)
        for r in tl.static_range(WORLD):
            # dont need to send myself anything
            if r != RANK:

                # tl.load(ll_otrs + r) is the start of r's buffers as mapped on this GPU
                # RANK * 2 * slot_words skips to my slot, slots are in rank order and each has
                #                       the two halves of our packed words
                # half:                 which half of our buffer are we writing to?
                peer = tl.load(ll_ptrs + r).to(tl.pointer_type(tl.int64)) + RANK * 2 * slot_words + half

                # write straight into peers memory
                tl.store(peer + o, word, mask=o<n_words)

    # Collect, spin on my own memory until every word from every peer carries this call's flag
    # then add in rank order in fp32
    for start in range(lo, hi, BLOCK):
        o = start + offs
        acc_lo = tl.zeros([BLOCK], dtype=tl.float32) # the low half of the words
        acc_hi = tl.zeros([BLOCK], dtype=tl.float32)

        for r in tl.static_range(WORLD):
            if r == RANK:
                # my own part never left this GPU just access
                d = _own_words(x32, o, o<n_words, row_words, TOPK, DT)
            else:
                # access until another ranks store lands here. volatile tells the compiler
                # to keep checking and not reuse the memory on every spin. 
                src = me + r * 2 * slot_words + half

                # other = flag is what masked lanes read as, the last chunk can be 
                # partly empty as a chunk is always BLOCK lanes but the message may
                # end part of the way through. The wait below checks all BLOCK of them
                # so they just have to look as though they arrived. To do this we fill
                # the invalid positions with the flag itself
                w = tl.load(src + o, mask=o<n_words, other=flag, volatile=True)

                # w >> 32 is each words flag (bit shifted over) and count the words whose
                # flag isnt C yet across the whole chunk
                spins = 0
                while (tl.sum(((w >> 32).to(tl.int32) != c).to(tl.int32), 0) > 0) & (spins < SPIN_LIMIT):
                    w = tl.load(src + o, mask=o < n_words, other=flag, volatile=True)
                    spins += 1
                if (spins >= SPIN_LIMIT) & (pid < 2):
                    tl.device_print("P2P SMALL ALL-REDUCE STUCK rank/block/call/words_missing", RANK, pid, c,
                                    tl.sum(((w >> 32).to(tl.int32) != c).to(tl.int32), 0))


                d = w.to(tl.int32) # int64 -> int32 only keeps the lower 32 bits, which is the packed data

                # the counter wraps (1, 2, ..., 2^30, 1, ...)
                # so an old word could pass as a new one so we zero it out as that is never possible
                tl.store(src + o, tl.zeros_like(w), mask=o < n_words)

            # Unpack the two values and accumulate
            d_lo, d_hi = _unpack(d, DT)
            acc_lo += d_lo
            acc_hi += d_hi

        # repack and write
        tl.store(out32 + o, _pack(acc_lo, acc_hi, DT), mask=o<n_words)


class P2PAllReduce: 
    """
    All-Reduce for tensor parallel on GPUs that can read/write on each other Memory
    """

    # Defaults if we dont _tune
    # BLOCK: How much of the message one block handles per pass of its loop
    # WARPS: How many groups of 32 threads run that block
    # GRID: How many blocks every call launches
    BLOCK, NUM_WARPS, GRID = 2048, 8, 32
    LL_BLOCK, LL_NUM_WARPS, LL_GRID = 128, 2, 64

    # Otherwise set a grid to try and pick the best config for a specific system. A triton warp
    # will have 32 threads so with w warps it has 32 * w threads. b (the block size) is how many words
    # the block processes per step, these are divided among threads. We ensure there is enough with b >= 32 * w.
    # for example, blok 64 with 4 warps (128 threads for 64 words) means there arent enough words for the allocated
    # threads and is waste
    CONFIGS = [(b, w) for b in (1024, 2048, 4096) for w in (4, 8) if b >= 32 * w]
    LL_CONFIGS = [(b, w) for b in (64, 128, 256, 512, 1024) for w in (1, 2, 4) if b >= 32 * w]
    GRIDS, LL_GRIDS = (16, 32, 64), (32, 64)

    def __init__(self, cpu_group, rank, world, max_elems, ll_max_elems = 64 * 1024, tune=True):
        """
        ll_max_elems: messages up to this many bf16 values use the flag-in-data kernel (0 = never)
        tune: time the candidate configs and keep the fastest (_tune, ~seconds), else the defaults
        cpu_group: a gloo process group, for the barriers. 
        rank / world: this GPU's rank and how many there are. 
        max_elems: the largest message (in values) that will ever be passed, which sizes the one-shot buffer
                   bigger ones should go to NCCL (see fits).

        Every rank must construct this at the same point: the allocations are collective.
        """

        self.rank = rank
        self.world = world
        self.max_elems = max_elems

        # The flag-in-data kernel works in words of 2 values, so ensure always even
        self.ll_max_elems = ll_max_elems - ll_max_elems % 2
        self.slot_words = max(self.ll_max_elems // 2, 1)

        # what group are we in?
        group = torch.distributed.group.WORLD.group_name

        def shared(n, dtype):
            """
            n values on every GPU, and each GPUs address as seen from mine
            https://docs.pytorch.org/docs/2.15/symmetric_memory.html
            """
            t = symm_mem.empty(n, dtype=dtype, device="cuda")
            ptrs = torch.tensor(list(symm_mem.rendezvous(t, group).buffer_ptrs), dtype=torch.int64, device="cuda")
            return t, ptrs

        # Allocate shared communication buffers, shared() allocates memory and returns
        # a local tensor and a list of the WORLD pointers, one for each GPU's copy
        # of the tensor mapped into this GPU's address space. We allocate enough memory
        # for the largest grid configuration that _tune might select, this way tuning
        # can change the grid size without needing to reallocate

        # The smallest SM count on any rank, no grid may be bigger than this
        sms = torch.tensor([torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count])
        torch.distributed.all_reduce(sms, op=torch.distributed.ReduceOp.MIN, group=cpu_group)
        sms = int(sms)

        # The candidate for _tune and defaults that fit
        self.configs = [(b, w, g) for g in sorted({g for g in self.GRIDS if g < sms} | {sms}) for b, w in self.CONFIGS]
        self.ll_configs = [(b, w, g) for g in sorted({g for g in self.LL_GRIDS if g < sms} | {sms}) for b, w in self.LL_CONFIGS]
        self.default = (self.BLOCK, self.NUM_WARPS, min(self.GRID, sms))
        self.ll_default = (self.LL_BLOCK, self.LL_NUM_WARPS, min(self.LL_GRID, sms))

        # largest grid size can be upto the SM count
        max_grid = max_ll_grid = sms

        # One shot kernel uses sync flags. The layout is [max_grid, WORLD], and each GPU has one flag
        # per (block, sender). flags[block, sender] stores teh call number that the sender has
        # finished writing for that block.
        self.flags, self.flag_ptrs = shared(max_grid * world, torch.int32)

        # the one shot buffer, each GPU writes its own partial sums here, and the other GPu reads them 
        # straight out of it over NVLink. It's 2 halves of max_elems values and call c uses half c % 2
        # (call 7 uses the half 1, call 8 uses half 0). 
        # We have two halves instead of one to do something known as double buffering. Once this GPU
        # has read its peer's data for call 7 it returns and moves on, but the peer may still be
        # reading this GPU's call 7 data... If call 8 wrote the same half, the peer would add a mix of
        # call 7 and 8 values which is wrong. Writing to the other half means call 8 cant touch what
        # the peer is still reading. A gpu cant finish call 8 until its peer has written call 8 which
        # can only happen AFTER it has finished call 7. Therefore we  allow for one GPU to be at most
        # one step ahead. The other option was to send done flags back and forth, but that costs
        # more communication which is more expensive than just having some more preallocated memory
        self.data, self.data_ptrs = shared(2 * max_elems, torch.bfloat16)

        # The low lateny path packs together a 4 byte synchronization call number along with 
        # 4 bytes of data, that contains 2 packed bf16/fp16 values. So each element is a total
        # of 8 bytes. The difference from earlier is peers PUSH into here, so it instead holds
        # other GPUs values, not mine. 
        self.ll_buffer, self.ll_ptrs = shared(2 * world * self.slot_words, torch.int64)

        # Each kernel's call per block count, used to know which half of a buffer to use
        self.counter = torch.zeros(max_grid, dtype=torch.int32, device="cuda")
        self.ll_counter = torch.zeros(max_ll_grid, dtype=torch.int32, device="cuda")

        # launch settings __call__ reads, it is the defaults until _tune() picks
        self.block, self.num_warps, self.grid = self.default
        self.ll_block, self.ll_num_warps, self.ll_grid = self.ll_default
        self.tuning = "defaults"    

        # Zero every buffer, flag and count on every rank, and nobody launches until all reset
        self._reset(cpu_group)
        if tune:
            self._tune(cpu_group)

    def _reset(self, cpu_group):
        """
        Bring every rank to a fresh state
        """
        torch.cuda.synchronize()
        torch.distributed.barrier(group=cpu_group)
        for t in (
            self.counter, self.ll_counter, self.flags, self.ll_buffer
        ):
            t.zero_()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=cpu_group)

    def _set(self, kernel, config):
        if kernel == "flag-in-data":
            self.ll_block, self.ll_num_warps, self.ll_grid = config
        else:
            self.block, self.num_warps, self.grid = config

    def _time(self, cpu_group, kernel, config, n, calls=20, reps=5):
        """
        Measure microsecond per all-reduce on bf16 values with a given configuration
        with CUDA Graphs enabled. Every rank must make the exact same set of
        calls, a kernel only finishes when its peers run it too
        """

        self._set(kernel, config)

        # New grid means new blocks, start every run from a fresh state
        self._reset(cpu_group)

        # just add zeros, doesnt matter its a benchmark
        x = torch.zeros(n, dtype=torch.bfloat16, device="cuda")

        # warmup run, this will trigger Tritons JIT compiler
        self(x)

        # Capture repeated calls into one CUDA Graph to amortize python launch
        # overhead and measure GPU kernel execution time under graph-replay conditions
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(calls):
                self(x)

        # warm up, run the replay once
        g.replay()

        # Every rank finishes the warm-up before either starts timing, otherwise one 
        # GPUs clock starts while the other is still busy
        torch.cuda.synchronize() # finish any outstanding GPU work
        torch.distributed.barrier(group=cpu_group) # ensure all participating rnks are ready 

        # time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = float("inf")

        for _ in range(3):
            start.record()
            for _ in range(reps):
                g.replay()
            end.record()
            end.synchronize()
            best = min(best, start.elapsed_time(end) * 1e3 / (reps * calls))
        
        return best

    def _tune(self, cpu_group):
        """
        Pick the block size, warps and grid for both kernels by timing every candidate on these GPUs
        """
        t0 = time.perf_counter()
        dist = torch.distributed

        # Every GPu has a fixed number of Streaming Multiprocessors (SM's) which then execute CUDA blocks
        # We limit the blocks to the smallest GPU's SM count so that all block can potentially run
        # at the same time. Imagine we have 20 SMs but schedule 40 things. Our GPU 0 could schedule
        # 0 - 20 waiting on GPU1 to also do 0 - 20. Lets say for some reason GPU1 schedules 5 - 25
        # isntead, well the overlap of 5 - 20 is good, but GPU0 will be stuck on 0 - 4 waiting for GPU1
        # and GPU1 will be stuck on 21-25 waiting on GPU0. So we cap it at SMs always. If we have different
        # gpus for some reason in one system, we cap to the smallest
        sms = torch.tensor([torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count])
        dist.all_reduce(sms, op=dist.ReduceOp.MIN, group=cpu_group)

        # the test sizes will be over the range of messages each kernel gets.
        ll_sizes = sorted(
            {   
                # n - n%2 always rounds down to an even number
                n - n % 2 for n in (
                    self.ll_max_elems // 64, self.ll_max_elems // 16, 
                    self.ll_max_elems // 4, self.ll_max_elems 
                ) if n >= 2
            }
        )   

        # one-shot benchmark of largeer messages, starting at atleast 64K
        lo = max(self.ll_max_elems, 64 * 1024)
        sizes = sorted(
            {
                n for n in (
                    min(k * lo, self.max_elems) for k in (2, 8, 32)
                ) if n > self.ll_max_elems
            }
        )

        # run tuming
        picked = []
        for kernel, configs, default, test_sizes in (
            (
                "flag-in-data", self.ll_configs, self.ll_default, ll_sizes
            ),
            (
                "one-shot", self.configs, self.default, sizes
            )
        ):
            configs = [c for c in configs if c[2] <= int(sms)]

            # if no test sizes, we keep defaults
            if not test_sizes:
                continue

            # each rank is timed on its own GPU, take the slowest
            us = torch.tensor(
                [
                    [self._time(cpu_group, kernel, c, n) for n in test_sizes]
                    for c in configs
                ]
            )
            dist.all_reduce(us, op=dist.ReduceOp.MAX, group=cpu_group)

            # How much slower than the best config averaged over sizes
            # 1.0 is the fastest at every size, 1.2 means 20% slower on average
            score = (us / us.min(0).values).mean(1)
            best = int(score.argmin())

            # keep the defaults unless the winner beats them by some percent (i picked 3%)
            # as anything inside 3% is probably run-to-run noise
            if default in configs and score[configs.index(default)] <= score[best] * 1.03:
                best = configs.index(default)

            self._set(kernel, configs[best])
            b, w, g = configs[best]
            picked.append(f"{kernel} block {b} warps {w} grid {g} "
                          f"({' / '.join(f'{t:.1f}' for t in us[best].tolist())} us at {test_sizes} values)")

        self._reset(cpu_group)
        self.tuning = "; ".join(picked) + f"; tuned in {time.perf_counter() - t0:.1f} s"

    @staticmethod
    def supported(world):
        try: # requires atleast torch 2.9 i think
            import torch.distributed._symmetric_memory
        except ImportError:
            return False
        return all(torch.cuda.can_device_access_peer(a, b) for a in range(world) for b in range(world) if a != b)

    def fits(self, x, topk=1):
        """
        Can we take this message? 2-byte floats only, contiguous, and a result no 
        bigger than our buffer. Anything else, the caller will just use NCCL
        """
        return (
            x.dtype in (torch.bfloat16, torch.float16) and x.is_contiguous()
            and x.numel() // topk <= self.max_elems
        )

    def __call__(self, x, topk=1):

        # x: [T * TOPK, H] where topk=1 for non MoE models
        row = x.shape[-1]

        # we can write to the input if not moe, otherwise we allocate the
        # shape of the reduced data passed in (after sum along experts)
        out = x if topk == 1 else x.new_empty(x.shape[0] // topk, row)

        # how many elements?
        n = out.numel()

        # just a sanity check
        if n == 0:
            return out

        # small bf16 messages (decode) will use flag-in data, one trip across the link
        if x.dtype in (torch.bfloat16, torch.float16) and row % 2 == 0 and n <= self.ll_max_elems:
            
            _p2p_ll_all_reduce_kernel[(self.ll_grid,)](
                x, out, self.ll_ptrs, self.ll_counter, n // 2, self.slot_words, row // 2, 
                RANK=self.rank, WORLD=self.world, BLOCK=self.ll_block, TOPK=topk, 
                num_warps=self.ll_num_warps)
            
            return out

        _p2p_all_reduce_kernel[(self.grid,)](
            x, out, self.data_ptrs, self.flag_ptrs, self.counter, n, self.max_elems,
            row, RANK=self.rank, WORLD=self.world, BLOCK=self.block, TOPK=topk, 
            num_warps=self.num_warps)
        
        return out

    def close(self):
        """
        Drop the shared buffers once no kernel will use them again
        """
        torch.cuda.synchronize()
        self.flags = self.data = self.ll_buffer = None

    def snapshot(self):
        """
        For debugging hangs
        """
        side = torch.cuda.Stream()
        out = {}
        with torch.cuda.stream(side):
            for name, t in (
                ("my_calls", self.counter[:self.grid]),
                ("my_small_calls", self.ll_counter[:self.ll_grid]),
                ("peer_flags", self.flags[:self.grid * self.world])
            ):
                out[name] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                out[name].copy_(t, non_blocking=True)
        side.synchronize()
        out = {k: v.tolist() for k, v in out.items()}

        # flags is [block, sender] flattened, so sender r's column is every WORLD-th entry from r
        flags = out.pop("peer_flags")
        for r in range(self.world):
            if r != self.rank:
                out[f"rank{r}_flags"] = flags[r::self.world]
        return out

def _time_all_reduce(fn, n, cpu_group, calls=20, reps=5):
    """
    Microseconds per all-reduce of n bf16 values
    """
    x = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    fn(x)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn(x)

    g.replay()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=cpu_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):
        start.record()
        for _ in range(reps):
            g.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / (reps * calls))
    return best

def _bench_rank(rank, world, port):

    dist = torch.distributed
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=world,
                            rank=rank, device_id=torch.device(f"cuda:{rank}"))
    cpu_group = dist.new_group(backend="gloo")
    sizes = [2 ** p for p in range(10, 23)]
    p2p = P2PAllReduce(cpu_group, rank, world, max_elems=sizes[-1])

    if rank == 0:
        print(p2p.tuning, flush=True)
        print(f"{'values':>9} {'KB':>7} {'torch NCCL us':>14} {'ours us':>8} {'faster':>7}  kernel", flush=True)

    for n in sizes:
        us = torch.tensor([
            _time_all_reduce(dist.all_reduce, n, cpu_group),
            _time_all_reduce(p2p, n, cpu_group),
        ])

        dist.all_reduce(us, op=dist.ReduceOp.MAX, group=cpu_group)

        if rank == 0:
            nccl, ours = us.tolist()
            kernel = "flag-in-data" if n <= p2p.ll_max_elems else "one-shot"
            print(f"{n:>9} {n * 2 / 1024:>7.0f} {nccl:>14.1f} {ours:>8.1f} {nccl / ours:>6.1f}x  {kernel}", flush=True)

    p2p.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    import socket
    world = torch.cuda.device_count()
    if world < 2 or not P2PAllReduce.supported(world):
        raise SystemExit("needs 2+ GPUs that can reach each other's memory")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    torch.multiprocessing.spawn(_bench_rank, args=(world, port), nprocs=world)
