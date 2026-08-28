# X post draft, GGUF-TP engine (v9)

Long-form single post (premium), first person, ~1,500 words. `---` marks where
it can split into a 4-part thread if preferred.

Note for posting: Twitter handles below verified where possible
(@vllm_project confirmed, @UnslothAI, @mastra, @antirez, @ggerganov,
@thsottiaux, @deepseek_ai), spot-check @datacurve and @pidotdev at post time.

---

The first DeepSeek V4 GGUF in vLLM.

DeepSeek V4 Flash is the first model I've run on my own hardware that felt
worth serious engineering (Qwen3.6-27B is also great, but didn't need any
engineering). The only way it fits on my 4× RTX 3090s is heavily quantized,
and luckily Antirez's IQ2_XXS GGUF (and Unsloth's UD_IQ1_M) keeps it good,
in my DeepSWE testing, indistinguishable from the API. But those weights
were stuck in llama.cpp, which could only use one GPU at a time. So I forked
vLLM and added a native GGUF execution path: a loader plus eight new CUDA
kernels that run the packed weights directly across all four cards, vLLM
Tensor Parallel style. For now, I'm calling it gguf-tp.

For the local-AI community, GGUF is the de-facto quantized-model format;
vLLM is the institutional-grade, high-throughput serving runtime
for safetensors. The gap between them, executing GGUF's packed low-bit
formats natively across GPUs, is what I closed for DeepSeek V4 Flash, on the
most common 24 GB consumer cards: the GGUF's packed bytes are executed
directly, no GGML, no requantization. Capped at 230 W / 1650 MHz to protect
my home circuits from overloading (ask me how I know, but there's 10–15%
more performance to be had by simply uncapping the power/clocks), and two
days of writing 8 new CUDA kernels.

The numbers: 76.7 tok/s single-stream decode, 551.9 tok/s cache-busted
prefill, 140K on-GPU context, and quality on par with DeepSeek via the API.
On DeepSWE it matched the DeepSeek API and the proven llama.cpp baseline,
but at 2.65× the speed. It scales nicely, too: the four cards saturate at
four concurrent requests, ~77 tok/s single stream, ~186 with four in flight,
and going to 8 doesn't buy much except latency at ~200 tok/s. llama.cpp
gives one thread ~38, so this is close to 5× the aggregate on the same
cards.

---

**The problem**

How good are the weights? In my DeepSWE testing, indistinguishable from the
API. Among the GGUFs I've tested that fit in 96 GB of VRAM, Antirez's
IQ2_XXS / Q2_K / Q8_0 family is the best balance of quality and size I
found, Unsloth's UD_IQ1_M edges it on token efficiency in some spots but
runs slightly bigger in GiB, which matters more than it sounds (you'll see
why when we get to context). The price: it uses about 10–25% more tokens
than the API in my testing, and it varies. But when the tokens are free on
your own machine, that's not as painful, as long as the quality holds.

But llama.cpp runs it at ~38 tok/s, a single request, because it can only
run one GPU at a time, on a box with four 3090s.

vLLM's own GGUF support doesn't help either: the standard path loads the
file, dequantizes the weights into normal floats, and runs those, no
packed-byte execution. And mine aren't even supported: IQ2 / Q2 aren't in
its tested set, and DeepSeek V4 isn't in its GGUF model list at all.
Officially it's "highly experimental" too.

And the obvious question, why not antirez's own DwarfStar engine, which
runs these exact bytes natively? It's Metal-first. Its CUDA multi-GPU design
is two tensor-parallel pairs, closer to DP=2 and TP=2 than TP=4, and the
residency math puts each card past 24 GiB before context is even reserved,
so it doesn't fit four 3090s. Close, but not this box. And it's a bespoke
engine, not vLLM.

My alternative was quantizing the safetensors down to this size so it could
run in vLLM with the Humming experts. I tried every way I knew. Every
attempt lobotomized the model and its quality, and I spent a small fortune
on H100 hours trying.

So I took a shot at true tensor parallelism with a GGUF on 4× 3090s, because
for the first time, this model felt worth the effort.

---

**Getting the bytes right**

The GGUF file stores weights in custom packed formats. My kernels had to
read those exact bytes and produce exactly the numbers llama.cpp produces,
otherwise the model quietly gets worse and I'd slowly go insane and I'd
never know why.

So before writing any kernel, I transcribed llama.cpp's decompression code
into one reference, and wrote a second, fully independent decompressor in
Python from the format spec. I ran 10,000 random samples plus adversarial
edge cases per format: byte-for-byte identical on all three. Painful, but
it caught a real bug in my independent decoder before it could drive me
insane later.

Then I mapped every tensor in the file, 1,328 tensors to 1,180 places in
the model, exact names, exact element counts, 21.19 GiB per GPU, zero
overlaps. That mapping is what proves no weight gets dropped, doubled, or
silently misplaced, and that the per-GPU footprint actually fits the cards.
The file hash is re-verified at load, and the low-bit weights are never
re-encoded, so that the GGUF stays my source of truth.

---

**The kernels**

I wrote eight CUDA C++ kernels from scratch, about 1,700 lines, in two
flavors: single-token decode and batched prefill.

The routed experts are the hard part. IQ2_XXS doesn't store weight values
directly, instead it packs lookup-table indices plus sign and sub-scale
bits into 66-byte blocks. A normal quantized format (like Q8_0) is a single
multiply per weight. Here, decoding a weight means: table lookup, unpacking
sign bits, computing the sub-scale, and folding all of it into the dot
product. The decode kernels read those blocks straight from the file and
compute with 8-bit integer instructions. The prefill kernels gather only
the experts the router picked and run them on the GPU's tensor cores.

Q2_K is the down-projection format: 4-bit scale nibbles plus 2-bit weights.
For prefill I fold the nibbles into the same 8-bit codes the tensor-core
instructions consume, so the scale math happens inside the matrix multiply
instead of a second pass. The down projection also stores its K and N axes
swapped relative to the up projection, so the kernel handles that too.

The attention and output projections are plain 8-bit (Q8_0), thank god, so
I did not need to write new matrix-multiply kernels for them. At load time I
convert them into the exact layout vLLM's battle-tested Marlin kernels
already eat, same numbers, just re-packed. Reusing the proven path kept
dense speed at vLLM's measured best, and avoided a float cache that would
have cost 100–120K of context.

One fused bonus: the gate-times-up weighting and the quantization of the
result fold into a single SwiGLU kernel, saving a pass per layer.

---

**Why I trust the tests**

Along the way I relied on a relatively small but rigorous battery of tests
to prevent boo boos. Tests can be sloppy and meaningless. I tried to be
rigorous and patient, for the sake of my future self. Each kernel is
checked against reference values, then replayed
from CUDA graphs, the fast path the server actually uses, and must give
identical results. Compute Sanitizer reports zero memory errors and zero
race conditions. The kernels were built capture-safe from day one: they take
the current stream and own their scratch memory, so they can't break the
graph path. It was slow and boring, and it's why I trust them.

---

**The numbers**

How I measured it: three warm-up runs, then five timed 512-token
code-generation responses (a Python B-tree request; temp 0.6 / top_p 0.95 /
top_k 20), 0.033% run-to-run spread. That single-stream 76.7 tok/s is 2×
llama.cpp's ~38 and equal to the old FP8 vLLM stack, on the Antirez GGUF
weights. Per layer on the 4-GPU graph: 0.193 ms decode; batched prefill
10.18 ms per layer at 256 tokens. The decode experts alone cost 42.2
µs/layer, beating the old WNA16 stack's ~50 µs (not that it matters because
that WNA16 quant was secretly mostly insane).

Prefill: 551.9 tok/s, cache-busted. My napkin math said 700+, so there's
probably more work to do here.

Context: 140K on-GPU, zero swap, and exact "needle in a haystack" recall at
119,730 prompt tokens. This is why I didn't pick the slightly bigger Unsloth
model, I just couldn't afford the context VRAM cost, tiny as it is.

---

**But did I murder it?**

The thing I care about most: quality. I'm not going to run a lobotomized
model just because it's fast.

On DeepSWE, it matches. Same task, same harness, max reasoning, one-cell
SuperJSON pilot. GGUF-TP: 0.9949 partial reward, 79/80 F2P, 116/116 P2P.
llama.cpp on the identical weights: 0.9898, 78/80, 116/116. And 2.65× faster
wall-clock, 42 minutes versus 111.

---

**So what's the catch?**

Prefill needs more work, as I mentioned above.

The context trade: llama.cpp can hold a single thread at a 430K context
window; this can hold 140K. Different trades, it wins on context, this wins
on speed. Plus I can start another quest to see if I can get 4-bit KV cache
working reliably with DSA and get some more breathing room.

One more catch, and maybe an opportunity: I compared every layer's output
against llama.cpp running the same weights. 28 of 43 layers sit outside the
tolerance I'd set in advance, based on napkin math, small differences that
compound as they flow through the network, worst around layer 20. I tested
the three most likely causes one at a time (how the router weights are
stored, the router's math precision, forcing a different expert execution
path). None of them is the single cause. The drift is the accumulation of
small, documented differences between the two engines: scale rounding when
the 8-bit weights get converted to Marlin's format, different addition
order in the integer math, a different attention kernel.

Critically though, the final outputs match, 0.9973 cosine similarity, same
top token, and the DeepSWE run showed zero behavioral difference. I'm going
to work on this drift later to see if anything interesting is hiding
underneath these rocks, but it definitely won't hold me back from running
this model on real tasks.

---

Code and research trail: https://github.com/Whamp/vllm, branch
`incubate/gguf-tp-sm86` (14 commits, tip `3ec20cebe`).

And I relied on and built on the work of:

- DeepSeek (@deepseek_ai, github.com/deepseek-ai/DeepSeek-V4-Flash), for the
  model
- vLLM (@vllm_project, github.com/vllm-project/vllm), the runtime this fork
  lives in
- haosdent (github.com/haosdent/vllm), whose DeepSeek V4 Ampere fork is the
  base everything here builds on
- llama.cpp (@ggerganov, github.com/ggml-org/llama.cpp), the reference for
  every byte contract and the quality baseline
- antirez (@antirez, github.com/antirez/ds4), for the GGUF this runs and the
  DwarfStar reference for the low-bit formats
- AppMana (github.com/AppMana/flash-mla), FlashMLA decode
- Unsloth (@UnslothAI, github.com/UnslothAI/unsloth), their UD GGUFs are
  excellent, just a few MiB too big in this case
- datacurve (@datacurve, github.com/datacurve-ai/deep-swe), the creators of
  DeepSWE
- @pidotdev, the DeepSWE harness where all of this was tested. If you think
  I did this all alone by hand, you're out of your mind.
- OpenAI (@openai), for GPT-5.6-sol, a lot of help, mostly on medium or
  high thinking. And @thsottiaux, Tibo does this earn a reset?
- Mastra (@mastra, github.com/Mastra-AI/mastra), this whole thing happened
  in one session across 2+ days and 76 compactions, only possible thanks to
  the pi-observational-memory extension in Pi
  (https://github.com/elpapi42/pi-observational-memory), which implements
  Observational Memory as invented by Mastra

---

Optional thread split if the single post is too long:
1. The hook + the situation (llama.cpp one GPU at a time, vLLM converts
   instead of executing, WNA16 requant lobotomized)
2. Getting the bytes right + the kernels
3. The tests + the numbers
4. But did I murder it? + one more catch + credits
