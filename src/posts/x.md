[Worklog] Training a 1B language model: Part 1

[REMOVE LATER] Ask Nishant Aklecha, Andrej Karpathy, Evan, Surya, Xavier, Faraz, Madhav Singhal, Shahbuland, Parth Sareen, Rajan Agarwal, Krish Shah. Any other cool people for review. Maybe someone from the toronto model club.

[Better intro] ???

While I have trained models in the past, none of them match this undertaking. My goal is to train a powerful 1B model for a particular down-stream task which I will not reveal atm and open-source evertything. This was inspired by [Nanochat](https://github.com/karpathy/nanochat) and [Olmo](https://allenai.org/olmo) but stems from a few personal takes.

I need the reader to note, these posts are primarily written as a streams of thoughts, that is, I write as I go and minimally edit it later. This is Part 1 of N and I plan to write out the basic training infra, train a simple llama3-like model on 8xH100s and evaluate how to step up.

Why am I going with ol’ llama3 for V0? because it has been reproduced many times by the community and is vanilla which helps set a baseline. Make no mistake, I am shooting for SOTA over the next few blogs.

Production systems don’t get made in one shot, for example, first we build a simple trainer, then mixed precision is introduced, and then keep 'em coming: 1D, 2D, 3D distributed training, fp8, custom kernels, fine-grained data mixing, blah blah.

Getting any model right has a lot of variables, but primarily comes down to data, architecture and compute. In this blog: for data, I will use Karpathy's [fine-web-edu-shuffled](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle). Over the next few blogs, I want to improve my training infra, curate my own datasets and make architectural changes to make it more inference friendly according to the final capability I want these models to possess.

I simply pre-train in this blog, mid- and post- training will be done down the line.

For data, I will use available high-quality open source datasets for all parts of training (pre-, mid-, post-). For architecture, I plan to start off with the simple llama3 arch. While I know methods which work on small models don’t necessarily work on big models as well, I need a way to test out architectural and data quality changes for cheap. For that I will ablate and train a tiny model one H100 over a few hours. Llama3 will be my baseline. Furthermore, All of this will be deterministically set through seeds. For the smaller runs I will have full reproducibility which slows the run down a bit but that’s fine as it gives us an accurate ablation.
```
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False
```

Why am I choosing 2048 seq len tokens? Initially, I wanted to directly go for 4096. Then, I remembered most dataset rows are <2k rows, so I'll need to collate many rows to make a 4096 token superrow. Furthermore, attention's quadratic nature is relevant in pre-training, 4096 token seq len will use 4x the compute as 2048 tokens.

Simple transformer training math: 131072*2048+32*(2048+2048*2048*2+2048*512*2+2048+3*2048*8192)+2048 (put pic)

A simple way to get that in torch is `sum(param.numel() for param in model.parameters())` = 1241581568. 

Cool, total memory = (activations + weights + gradients + optimizer state) * 4 bytes per param + misc. 

Misc like kernels, rope cache, etc is 5GB.

weights = 1241581568 params
grads = 1241581568 params
activations = dependant on implementation (can be brought down with good kernels) lets say n * 1241581568. Empirically on a non-compiled model I get n around 1.6 for (1, 2048) input. it scales linearly (I confirmed this empirically with an experiment) with batch size.
optimizer = first moment + second moment = 2 * 1241581568.

However, simply adding these up won’t give peak memory as different stages store different tensors. This is the basic training process:
```
loss = model.forward(tokens[:, :-1], targets=tokens[:, 1:])
loss.backward()
optim.step()
```

Stepping through this, model.forward stores the weights + activations (+ grad if using gradient accum which we will later). backward() calculates gradients using activations and weights. optim.step() updates weights using optimizer states and grads. I tried a first pass with torch.compile (granted i ran it on my mac which doesn’t have cuda-level compile reductions), a very quick manual profile and it didn’t help with the memory, kernels it is but that comes later on H100s. [THIS IS WRONG, IN STEADY STATE grads, weights and optimizer states are stored all the time]

With the math shown above, we show that when training on > 1 batches, peak memory is used in loss.backward() which = weights + grads + activations. Actually, real life is more complicated. In steady state (after first step), optimizer states always need to be stored and if using grad accumulation, grads need to be preserved over steps. Which means real peak memory in the run = weights + activations + grads + optimizer states for local activations.

Activation memory when inputs are [N > 1, 2048] is astronomical, there is a known path to reducing activation memory: torch.compile and then, CUDA kernels for missed optimizations. Torch stores activations after most of it’s ops to calculate backwards, however, a lot of these ops can be fused, eliminating these intermediate activations!

Let’s talk about token budget (total tokens to train on). Scaling laws say a 1:20 ratio of params:tokens is optimal. I have 1B params so 20B tokens is what I need to train on. My target batch size is 1M tokens which I chose because GPT-3 XL was trained on 1M (2^20) batch size, so I chose it pic ![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/03/OpenAI-GPT-3.png). This means I will have 20000 steps to train my model through. Now, let’s do some maths with the naive non-compiled numbers above, I have 8xH100’s meaning 80gb per gpu. Training through 1 batch of 2048 tokens takes 5.6 * 1241581568 * 4 in FP32 memory which is 25.9GB of memory maximum. [N, 2048] tokens should take (4 + N * 1.6) * 1241581568 * 4 in FP32 memory which is 18.6GB + N * 7.3GB. On one 80GB H100 (accounting 5GB misc for kernels, etc), ~7 batches of 2048 tokens will fit given my basic calculations!

2^20 tokens (1,048,576) batch size is an input of shape [512, 2048] tokens. 8 GPUs doing [7, 2048] tokens at a time ([56, 2048] tokens being done at a time) means training one global batch needs to accumulate gradients over ceil(256/56) = 5 steps. Total steps needed to train the model is (20B / 1M) * 5 = 20000 * 5 = 100,000 steps

Remember this is without any optimizations. But, I have a few things up my sleeve which don't affect accuracy: torch compile, flashattn-3, gradient accumulation, Mixed precision w/ BF16, etc. Worst case, we can even move to FSDP but that will add more communications overhead (this could be eliminated) compared to DDP, anywho i will benchmark all of this later anyway.

I ran my calcs on H100 this time and the activation memory on a [1, 2048] tokens input is astronomically more. On my mac it was 6.2GB and on my H100 it is 22.5GB????? I need to investigate this. My first thought is different op path used on H100 means more activations are saved.

To debug this I spoke with my confidants claude and chadgpt. Claude was more helpful and was a better brainstorming partner. After brainstorming debugging ideas and me spending time with the debugger to tick things off, we landed on and cowrote a script which calculated the memory diff as we forward through each layer and there lied the smoking gun: H100 attention activations per layer was OOMs more compared to CPU attention activations. Upon deeper inspection turns out this was because CUDA SDPA with FP32 inputs falls back to naive attention (MATH backend) which materializes the full attention logit matrix. I confirmed this by casting QKV to bf16 and re-casting the output to fp32 to see 0.1GB in activations.

Let's say we move everything to BF16 from now on. With fp32, we could only fit [7, 2048] on each GPU. Now, we should be able to do [14, 2048] which brings accumulate steps to 3 (2.28 to be precise, some compute will be wasted each batch, let's see what we can do about this)!

## Optimization

Let's get the fun part done first. In most LLM training runs, engineers have some goals related to optimization: maximize MFU (mean flops utilization), maximize training throughput, minimize communication overhead (MAYBE MORE). Let's get it.

The first thing to do is to stand on the shoulder of giants and run torch compile. I wrote a very simple script for this section to test out optimization changes. 
```
import time
import torch
from torch.optim.adamw import AdamW
from src.model import LLM
from src.utils import ModelConfig

device = torch.device('cuda')
config = ModelConfig()
model = LLM(config).to(device)
model.init_weights()
optimizer = AdamW(model.parameters())

for i in range(5):
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    tokens = torch.randint(0, 131072, (4, 2048), dtype=torch.int64).to(device)
    loss = model.forward(tokens, targets=tokens)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"Step #{i} | Time: {end - start:.3f} s | Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

This is my base per-training-step stats (time per step + peak mem per step) with [1, 2048] fp32 input on the H100:
```
Step #0 | Time: 0.705 s | Allocated: 14.97 GB | Peak: 24.90 GB | Reserved: 25.16 GB
Step #1 | Time: 0.443 s | Allocated: 14.97 GB | Peak: 33.26 GB | Reserved: 34.02 GB
Step #2 | Time: 0.443 s | Allocated: 14.97 GB | Peak: 33.26 GB | Reserved: 34.02 GB
Step #3 | Time: 0.443 s | Allocated: 14.97 GB | Peak: 33.26 GB | Reserved: 34.02 GB
Step #4 | Time: 0.443 s | Allocated: 14.97 GB | Peak: 33.26 GB | Reserved: 34.02 GB
```

Let's run basic torch compile `model = torch.compile(model, mode='reduce-overhead', fullgraph=True)` and see the results:
```
Step #0 | Time: 4.463 s | Allocated: 14.90 GB | Peak: 24.83 GB | Reserved: 37.92 GB
Step #1 | Time: 0.525 s | Allocated: 14.90 GB | Peak: 30.54 GB | Reserved: 37.92 GB
Step #2 | Time: 0.406 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 37.92 GB
Step #3 | Time: 0.405 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 37.92 GB
Step #4 | Time: 0.405 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 37.92 GB
```

Slight increase in reserved memory with a 10% drop in time per step (empirically, reserved memory goes down and time/step consistently drops -10% over larger batches when using torch compile). Note: reserved memory will be our main memory metric.

Let me explain quickly how this works. Normally, when running pytorch, the operations are "lowered" into C++/CUDA ops that form a computational graph which then runs on your target device. When running these ops on GPUs, a cuda kernel is invoked per graph node and there are launch overheads associated with each invocation. On a profiling chart, these look like empty spaces between each kernel running (referred to as "bubbles"). [ADD PIC OF BUBBLES] [MAYBE WRITE A BLOG ON THIS]

Torch compile fuses most ops into one kernel instead which removes these bubbles and memory access overheads and forms an optimized computational graph (CUDA graphs) which runs FAST.

### Mixed Precision
I haven't mastered the dark arts of low-precision training yet. H100's have support for fp8 but I will stick to bf16 for this time. In Pytorch, explicit casting is setting the whole model to be the target type (bf16 in our case) which also sets the optimizer states to bf16. BF16 has a high range but much lower precision compared to fp32 which could lead to convergence issues. Therefore, we will use torch's autocast which stores master weights and optimizer states in fp32, but casts weights and inputs to bf16 for forward and backward passes which means the activations are  bf16 (max 50% savings in activation memory).

Making this change is simple, I only have to add autocast to the forward pass:
```
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(tokens, targets=tokens)
```

```
Step #0 | Time: 22.598 s | Allocated: 14.90 GB | Peak: 24.83 GB | Reserved: 30.87 GB
Step #1 | Time: 0.244 s | Allocated: 14.90 GB | Peak: 24.83 GB | Reserved: 31.94 GB
Step #2 | Time: 0.077 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 31.94 GB
Step #3 | Time: 0.077 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 31.94 GB
Step #4 | Time: 0.077 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 31.94 GB
```

Reserved memory also drops and actually doesn't go up much as we increase batch size which means this will let us reduce the number of accum steps by a lot.

### Gradient Checkpointing
We reduced activation memory by cutting it's byte size by half (fp32 -> bf16). Now, we will explore gradient checkpointing which will not store some activations during the forward pass and recalculate them during the backward pass when needed. This should let us really cut back on activation memory needed.

Before we do this, let's measure the time per op and memory per op. 

We will review the profiling chart later to see if there are any missed opportunities for fusion in the training steps.

If I can minimize the number of grad accum steps, that would be awesome. My idea for this is to introduce gradient checkpointing which skips storing some marked operations for the backward pass and gets recalculated later on.

Till before gradient checkpointing, with the optimizations we discussed above, I can easily train on [16, 2048] tokens. I get these results:
```
Step #0 | Time: 9.873 s | Allocated: 14.90 GB | Peak: 63.12 GB | Reserved: 78.59 GB
Step #1 | Time: 0.625 s | Allocated: 14.90 GB | Peak: 73.06 GB | Reserved: 78.59 GB
Step #2 | Time: 0.456 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 78.59 GB
Step #3 | Time: 0.457 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 78.59 GB
Step #4 | Time: 0.458 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 78.59 GB
```

Once I add gradient checkpointing on every second layer, I get these results:
```
Step #0 | Time: 29.018 s | Allocated: 14.90 GB | Peak: 39.16 GB | Reserved: 54.63 GB
Step #1 | Time: 0.617 s | Allocated: 14.90 GB | Peak: 49.10 GB | Reserved: 54.63 GB
Step #2 | Time: 0.509 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 54.63 GB
Step #3 | Time: 0.506 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 54.63 GB
Step #4 | Time: 0.506 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 54.63 GB
```

A 12% increase in compute time with a 24GB drop in reserved memory which is nuts. 
3
Let's try checkpointing all layers:
```
Step #0 | Time: 30.168 s | Allocated: 14.90 GB | Peak: 24.83 GB | Reserved: 34.63 GB
Step #1 | Time: 0.606 s | Allocated: 14.90 GB | Peak: 29.03 GB | Reserved: 34.63 GB
Step #2 | Time: 0.534 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 34.63 GB
Step #3 | Time: 0.532 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 34.63 GB
Step #4 | Time: 0.533 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 34.63 GB
```

Which is a 17% increase in compute time for a 44GB drop in reserved memory. Now, we can double the input batch size to [32, 2048] and see the results:
```
Step #0 | Time: 36.246 s | Allocated: 14.90 GB | Peak: 32.11 GB | Reserved: 49.33 GB
Step #1 | Time: 1.096 s | Allocated: 14.90 GB | Peak: 42.05 GB | Reserved: 49.33 GB
Step #2 | Time: 1.019 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 49.33 GB
Step #3 | Time: 1.018 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 49.33 GB
Step #4 | Time: 1.020 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 49.33 GB
```

Seems like we still have some VRAM space to fill, let's try an input of [64, 2048] and hope it doesn't OOM.
```
Step #0 | Time: 51.364 s | Allocated: 14.90 GB | Peak: 58.15 GB | Reserved: 78.78 GB
Step #1 | Time: 2.139 s | Allocated: 14.90 GB | Peak: 68.09 GB | Reserved: 78.78 GB
Step #2 | Time: 2.010 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 78.78 GB
Step #3 | Time: 2.010 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 78.78 GB
Step #4 | Time: 2.005 s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 78.78 GB
```

I think this is perfect because I don't need gradient accumulation anymore. There are still a few optimizations left on the table.

I switched to fused AdamW which should use a more efficient optimizer implementation that helps save some more memory. Now, reserved memory drops down to 73GB.

I tried switching to allowing the [tf32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) which stores everything in fp32 as we currently have it but does all the math with 19 bit floats (e8m10) on special tf32 compute units which should be faster. However, I did not notice any improvements in speed in my scratchpad tests. FYI, I used `torch.set_float32_matmul_precision("high")`.

All the changes I have until now should not affect numerical stability and that is pretty important.

Pytorch uses Flash Attention 2 (FA-2) which is fast but [Flash Attention 3](https://pytorch.org/blog/flashattention-3/) is purpose-written for hopper chips and is 50% faster than FA-2. The only problem here is FA-3 needs to be compiled from souce and is infamous for it's loooong compile times. My [process](https://pypi.org/project/flash-attn/2.8.3/) to get FA-3 is:
1. `uv add packaging ninja`
2. `git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention`
3. `cd hopper && python setup.py install`
4. Use flash_attn_interface.flash_attn_fun() instead of SDPA

Once FA-3 compiled, I tried it out in my code but now I get an OOM. FA-3 introduces a graph break in my LLM module which might add some inefficiency. So I reduced batch size to 8 and noticed that it takes more time per step and memory than SDPA's FA-2 implementation. The conclusion is there is definitely a bug in my setup. I might revisit this again to see what's wrong.

<Check COMMUNICATION OVERHEAD OVER MANY DEVICEs>

Do a graph of different operations selectively checkpointed vs speed and memory savings.

## Data planning

Data planning is probably the most important part of the model training process. Garbage in, Garbage out.

I have two goals for the end model:
* Mid-context length ready: 90% of the steps will have an input with sequence length 2048 tokens and the rest 4096 tokens.
* Qualitative: Good at english, math and code tasks for a particular downstream task.

I will need to implement fine-grained data-mixing for this. I have a 20B token budget. 90% of that (or 18B tokens) will be 2048 token seq len, let's talk about that first. To have excellent english understanding, I will assign 50% of data to be from huggingface's filtered [fineweb-edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-10BT/train?p=2&row=290), 20% will be HF's [finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki), 20% code from [Star coder](https://huggingface.co/bigcode/starcoder) and the last 10% will be from [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math).

On second thought, I want to ensure my training codebase works first and so, I will simply train on Karpathy's [fine-web-edu-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle).

## General training infra

* checkpointing logic
* logging/metrics
* learning rate schedules
* warmup strategy

Write about this later.

## Battling the training overlords

I expected to face some turbulence during training. I have read multiple great blogs about training convergence issues like the famous [Marin 32B's loss spikes](https://wandb.ai/marin-community/marin/reports/Marin-32B-Spike-Fixing--VmlldzoxMzM1Mzk1NQ) blog.

Because training on 8xH100's directly without verifying the run is a catastrophical waste of money, I rolled out this run in 3 stages. First stage is local device testing where I ran training on my Macbook on a much smaller model. This was to identify any general pytorch related bugs that I might have introduced. Once I solved those, I moved on to running it on 1xH100 where I squashed a few minor torch.distributed cleanups and cuda-related errors. I noticed over 250 steps that my loss was down and to the right which made me smile. The loss starts of as ~12 and -ln(1/131072) is 11.78, since the model has no knowledge, the probability of getting any token is 1/vocab_size and therefore, cross entropy loss is -ln(1/131072), so we are bang on the expected first loss value. 

With [64, 2048] inputs on the 1xH100 and over 256 steps, the model went from a loss of 12.26 to 6.18. To get a better idea of how my actual training run will do, I set grad_accum_steps to 8 (effective batch size = [512, 2048] tokens) which is how I want to run my actual training.

Each step of [64, 2048] takes ~2s on the H100, so going over 250 steps is not a challenge. However, convergence issues usually come in the middle and end of the run, so it makes more sense to start an 8xH100 run and carefully monitor it instead. I also combed over bugs using GPT5 and Claude, so I don't have any surprises later. 

That's when I hit a snag. Torch.compile + gradient accumulation has some quirks, since I run backward() without runnning optimizer each step, I get `Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`. I spent a bunch of time trying different things like the suggested `torch.compiler.cudagraph_mark_step_begin` to no vain. Finally, my idea was to disable CUDA graphs all together (maybe bad idea in the long term?) and use the 'default' torch.compile mode and it worked! Now, my problem is with higher batch size, reserved memory explodes (reduce-overhead mode was saving me all this time). You know what, I don't even need grad accum for now so this is not a pressing problem, let's move on to 2xH100 with the proper torch compile and no grad accum.

I wanted to run this on a 2xH100 to minimize costs if I end up hitting a snag and debugging, while still checking for correctness on a distributed setup. My first mistake with this run was guessing the order of applying ddp and compile, note to always apply `model = DDP(model, ...)` first and then `model = torch.compile(model, ...)`. Once that was sorted, the run failed again because of another CUDA-graph related problem. I decided to try and relax the graph constraint by turning `fullgraph=False`. If that did not work, I would have made torch compile mode='default' again but I really don't enjoy the increase in reserved memory (and subsequent OOMs) it causes. torchrun is used for this.

Finally, it was time to run on the 8xH100 cluster. It works from the getgo! 

A problem I had was around 15k steps my loss was stagnant. My intuition was that the learning rate is too low, there was no way for me to verify this given I was not logging things like ||update|| / ||param|| which would have helped me debug this more. So, I decided to stop training, loaded the 15k checkpoint to be trained on 5k more steps but this time with a different schedule (1.5*base_lr, very low warmup steps, higher minimum lr ratio) which should allow the model to break out of a local minima. This should be reflected in this wandb graph. A minor benefit of this is it allowed me to verify me checkpointing logic (although I should have tested it on 1xH100 first) and fix all the minor bugs (if any were major I would have to start all over again). I don't enjoy YOLO runs and this one did not work.

This is my graph for the first 15,000 steps:

!()[/Users/omkaarwork/Desktop/projects/personal-website/public/x-worklog-1b-llama/8xh100_run.png]

Overall, this was fun, however my MFU was at 18% & train throughput was at 400k tok/s which is no bueno. I know my checkpointing on every layer impacted throughput a lot, however it allows me to skip gradient accumulation, so I need to think a bit more about this tradeoff. As part of my explorations, I have to train more models. I want to minimize cost as much as possible while making my performance gains translatable to all kinds of future runs. I will work on this over next post.

I have a few reflections. I started off with a massive scope (which I ommited here) but realized it's too much for a fresh codebase. I wrote a very bad but working first iteration, then refined things as I cut down requirements which reminds me of this picture:
!(SpaceX's thruster optimization)[https://i.ytimg.com/vi/wEMMqhVMXuQ/maxresdefault.jpg]

It's much easier to iterate from bad to good then land up on good straight away. Don't get me wrong me repo is still far from SOTA but it provides a clear working implementation which I can abstract further later. That brings us to my second reflection, I had to rely on other's observations for learning rate, weight inits, scheduler and other hyperparams. Great researchers have a mix of empirical evidence along with mathematical intution for what works based on 1000s of past runs. It's something that I'll just have to iterate and read more papers on.

### Extras

Tim Dettmers has a library called bitsandbytes which is widely used for quantization. They have an nf4 format which is widely used in QLoRA finetuning of models, it stores all parameters in 4-bit and uses 2 block scaling factors to dequantize them. Check this [paper](https://arxiv.org/abs/2305.14314) to learn more. Anyway, they have an AdamW8bit optimizer which is commonly used in finetuning. I brought it into my 1xH100 training run along with StableEmbedding which is the recommended setting. My reserved memory dropped by 10GB and these are the two graphs:

_____________________________ END __________________________________

Things to remember to inlude in blog:
* Benchmark DDP vs FSDP-2
* 90% of run is on 2048 tokens ctx, 10% will be on 4096 tokens


# Next steps
- data mixing
- cross doc masking
- MLA
- FA-3
- fp8
- Muon
- MOE
- SWA
- QK Norm
- deepseek's multi token prediction


Help me think through all of this by looking stuff up. I need everything to be backed by science and industry best practices.



checkpointing - i want to checkpoint the model locally every 100 steps and every 1000 steps to a registry. 

logging metrics - wandb for logging every 10 steps (maybe 50?). I want to setup 