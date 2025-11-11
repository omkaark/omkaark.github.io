# [WORKLOG] Training a 1B model - Part 2

There's a few things I wanted to note up top:
- I used reserved memory as my memory metric because torch.compile is causing wierd values for other memory metrics


## Performance Optimizations

This will be an optimization heavy blog. I want to make my run cheaper / converge faster, to be very honest, a few percent savings at this scale does not matter... but I do this as "love for the game". I wanted to tweak architecture and test out more data/hyperparam ideas but that should come after I make infra optimizations. These optimizations will aim to reduce memory usage and increase training throughput. I have a few ideas for this:

### Flash Attention 3
Throughput maxxing

In the last post, we ommited FA3 and went for standard torch SDPA. I tried really hard to get it to work, I spent hours on the debugger, pleaded it to work, flipped some settings, etc. and I think I know the problem.

It's definitely not an autocast thing. I think it's a torch compile problem.

Without torch.compile, these are the first three train steps with and without FA3 on an [8, 2048] input.

```
Using Flash Attention 3
Step #0 | Time: 0.596s | Allocated: 14.97 GB | Peak: 29.36 GB | Reserved: 36.52 GB
Step #1 | Time: 0.407s | Allocated: 14.97 GB | Peak: 39.33 GB | Reserved: 53.70 GB
Step #2 | Time: 0.406s | Allocated: 14.97 GB | Peak: 39.33 GB | Reserved: 53.70 GB

Using Pytorch SDPA
Step #0 | Time: 0.604s | Allocated: 14.97 GB | Peak: 29.36 GB | Reserved: 36.52 GB
Step #1 | Time: 0.428s | Allocated: 14.97 GB | Peak: 39.33 GB | Reserved: 53.70 GB
Step #2 | Time: 0.427s | Allocated: 14.97 GB | Peak: 39.33 GB | Reserved: 53.70 GB
```

Notice same memory usage while FA3 being about 3.4% faster. Now, this is with torch compile (fullgraph=False as FA3 is not compilable [yet](https://github.com/Dao-AILab/flash-attention/pull/1769)).

```
Flash Attention 3
Step #0 | Time: 2.035s | Allocated: 7.48 GB | Peak: 25.20 GB | Reserved: 29.57 GB
Step #1 | Time: 0.390s | Allocated: 7.48 GB | Peak: 30.17 GB | Reserved: 16.79 GB
Step #2 | Time: 0.360s | Allocated: 7.52 GB | Peak: 30.07 GB | Reserved: 38.26 GB
Step #3 | Time: 0.364s | Allocated: 7.52 GB | Peak: 30.07 GB | Reserved: 38.26 GB
Step #4 | Time: 0.358s | Allocated: 7.52 GB | Peak: 30.07 GB | Reserved: 38.26 GB

Pytorch SDPA
Step #0 | Time: 30.302s | Allocated: 7.45 GB | Peak: 11.59 GB | Reserved: 18.23 GB
Step #1 | Time: 0.339s | Allocated: 7.45 GB | Peak: 16.56 GB | Reserved: 18.23 GB
Step #2 | Time: 0.294s | Allocated: 7.45 GB | Peak: 7.45 GB | Reserved: 18.23 GB
Step #3 | Time: 0.301s | Allocated: 7.45 GB | Peak: 7.45 GB | Reserved: 18.23 GB
Step #4 | Time: 0.309s | Allocated: 7.45 GB | Peak: 7.45 GB | Reserved: 18.23 GB
```

SDPA is compile-compatible so torch can reserve memory realistically. With FA3, since it's a black box op and not torch compile supported, it reserves more memory. Sadly, the conclusion is to stick with SDPA since it's faster when compiled and consumes less memory. We might have to revisit FA3 if our FP8 experiments end up working (low-precision attention is the main selling point in general).

Note: If you are an expert and think my conclusion is incorrect, I'd love to heard from you on my socials!

### Gradient Checkpointing
We checkpointed the whole LLM in the last blog. Right now, I want to experiment and understand the compute-memory tradeoff of different ops with a [16, 2048] input. So, I will be adding checkpointing to different ops in this and measuring perf:
```
class CheckpointedDecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.attention = Attention(config)
        self.swiglu_norm = RMSNorm(config)
        self.swiglu = SwiGLU(config)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        norm = self.attn_norm(x)
        attn = self.attention(norm, cos, sin)
        x = x + attn
        norm = self.swiglu_norm(x)
        swiglu = self.swiglu(norm)
        x = x + swiglu
        return x
```

Here are different parts of the transformer checkpointed along with compute time vs memory consumed relative to baseline.

`Baseline`: <+0% Compute, +0% Memory
```
Step #2 | Input Size: [16, 2049] | Time: 0.500s | Reserved Memory: 71.48 GB | Loss: 12.257
Step #3 | Input Size: [16, 2049] | Time: 0.519s | Reserved Memory: 71.48 GB | Loss: 12.253
Step #4 | Input Size: [16, 2049] | Time: 0.503s | Reserved Memory: 71.48 GB | Loss: 12.254
```

`Checkpointed: Embedding`: 0% Compute,  -0.5% Memory
```
Step #2 | Input Size: [16, 2049] | Time: 0.497s | Reserved Memory: 71.10 GB | Loss: 12.263
Step #3 | Input Size: [16, 2049] | Time: 0.525s | Reserved Memory: 71.10 GB | Loss: 12.250
Step #4 | Input Size: [16, 2049] | Time: 0.499s | Reserved Memory: 71.10 GB | Loss: 12.257
```

`Checkpointed: Attn and SwiGLU norm`: +0.1% Compute, -0.5% Memory
```
Step #2 | Input Size: [16, 2049] | Time: 0.500s | Reserved Memory: 71.48 GB | Loss: 12.258
Step #3 | Input Size: [16, 2049] | Time: 0.521s | Reserved Memory: 71.48 GB | Loss: 12.249
Step #4 | Input Size: [16, 2049] | Time: 0.504s | Reserved Memory: 71.48 GB | Loss: 12.252
```

`Checkpointed: Attention`: +6% Compute, -11% Memory
```
Step #2 | Input Size: [16, 2049] | Time: 0.540s | Reserved Memory: 63.53 GB | Loss: 12.258
Step #3 | Input Size: [16, 2049] | Time: 0.554s | Reserved Memory: 63.53 GB | Loss: 12.253
Step #4 | Input Size: [16, 2049] | Time: 0.535s | Reserved Memory: 63.53 GB | Loss: 12.247
```

`Checkpointed: SwiGLU`: +5.65% Compute, -40% Memory
```
Step #2 | Input Size: [16, 2049] | Time: 0.568s | Reserved Memory: 42.56 GB | Loss: 12.261
Step #3 | Input Size: [16, 2049] | Time: 0.557s | Reserved Memory: 42.56 GB | Loss: 12.252
Step #4 | Input Size: [16, 2049] | Time: 0.572s | Reserved Memory: 42.56 GB | Loss: 12.260
```

`Checkpointed: Decoder Block`: +17% Compute, -56% Memory
```
Step #2 | Input Size: [16, 2049] | Time: 0.602s | Reserved Memory: 29.67 GB | Loss: 12.251
Step #3 | Input Size: [16, 2049] | Time: 0.601s | Reserved Memory: 29.67 GB | Loss: 12.251
Step #4 | Input Size: [16, 2049] | Time: 0.609s | Reserved Memory: 29.67 GB | Loss: 12.268
```

To be fair, full gradient checkpointing does not hit MFU as bad as I thought... only about 12%. I think checkpointing on SwiGLU is the move here. If I can reduce memory usage in other areas, I can reduce the amount of checkpointing I do and get a higher MFU!!

### Eliminate logits
Memory minning

Fused Chunked Cross Entropy variants are getting popular now, they basically drastically reduce the memory needed during cross entropy loss calcs by not needing to materialize logits upfront. I already wrote blogs on one such variant - [Cut Cross Entropy by Apple](https://omkaark.com/posts/cce.html) - and [my failed experiment with it](https://omkaark.com/posts/cce-experiment.html). 

#### Liger Kernels
Linkedin's Liger Kernels gained popularity as well for their Fused Linear Cross Entropy implementation which touts big perf gains.

There's not a huge number of reference implementations out in the wild for this, so I took whatever is in their Readme, read their paper and repo code, and adapted it to my code.

I replace 
```
logits = self.lm_head(x)
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.shape[-1]), 
    targets.reshape(-1)
)
```
with:
```
In LLM.__init__():
self.lce = LigerFusedLinearCrossEntropyLoss()

In LLM.forward():
x_flat = x.reshape(-1, x.shape[-1])
targets_flat = targets.reshape(-1)

loss = self.lce(
    self.lm_head.weight,
    x_flat,
    targets_flat
)
```

I noticed something wierd, these results are with torch.compile (fullgraph=False because liger kernel has graph breaks on .item()) on [32, 2048] inputs.
```
Linear Cross Entropy
Step #0 | Time: 53.53s | Allocated: 14.90 GB | Peak: 45.54 GB | Reserved: 66.24 GB
Step #1 | Time: 0.700s | Allocated: 14.90 GB | Peak: 55.47 GB | Reserved: 66.24 GB
Step #2 | Time: 0.654s | Allocated: 14.90 GB | Peak: 14.90 GB | Reserved: 66.24 GB

Regular Cross Entropy
Step #0 | Time: 34.526s | Allocated: 14.90 GB | Peak: 19.87 GB | Reserved: 29.59 GB
Step #1 | Time: 0.652s | Allocated: 14.90 GB | Peak: 29.03 GB | Reserved: 29.59 GB
Step #2 | Time: 0.610s | Allocated: 14.90 GB | Peak: 14.90 GB | Reserved: 29.59 GB
```

Torch compiler asked me to add `torch._dynamo.config.capture_scalar_outputs = True` and I got 
```
Linear Cross Entropy
Step #0 | Time: 1.426s | Allocated: 14.97 GB | Peak: 19.93 GB | Reserved: 19.96 GB
Step #1 | Time: 0.936s | Allocated: 14.97 GB | Peak: 25.37 GB | Reserved: 26.54 GB
Step #2 | Time: 0.937s | Allocated: 14.97 GB | Peak: 25.37 GB | Reserved: 26.54 GB
```

What on god's green earth did I just witness... LCE is slower and hogs more memory? So, I turn torch compile off and ran the bench again on [16, 2048] token inputs.
```
Linear Cross Entropy
Step #0 | Input Size: [16, 2048] | Time: 1.440s | Reserved: 19.96 GB
Step #1 | Input Size: [16, 2048] | Time: 0.939s | Reserved: 26.54 GB
Step #2 | Input Size: [16, 2048] | Time: 0.943s | Reserved: 26.54 GB

Regular Cross Entropy
Step #0 | Input Size: [16, 2048] | Time: 0.992s | Reserved: 65.35 GB
Step #1 | Input Size: [16, 2048] | Time: 0.825s | Reserved: 82.53 GB
Step #2 | Input Size: [16, 2048] | Time: 0.828s | Reserved: 82.53 GB
```

It's still slower but consumes MUCH less memory (sus). I'd expect it to consume 16*2048*131072*2 bytes (or 8GB) less in activation memory. I honestly do not know what to make of these results. For now, let's skip this and try apple's cut cross entropy.

#### Cut Cross Entropy

I am so close to rolling out my own LCE kernel, but I want to exhaust all the ez options first. Cut Cross entropy is by Apple, these are results with torch compile with [64, 2048]:
```
Cut Cross Entropy (Vanilla)
Step #0 | Batch Size: (64, 2048) | Time: 34.40s | Reserved Memory: 57.62 GB | Loss: 11.10
Step #1 | Batch Size: (64, 2048) | Time: 2.844s | Reserved Memory: 58.69 GB | Loss: 9.118
Step #2 | Batch Size: (64, 2048) | Time: 2.811s | Reserved Memory: 59.76 GB | Loss: 7.811

Cut Cross Entropy (FP32 Accum)
Step #0 | Batch Size: (64, 2048) | Time: 36.084s | Reserved Memory: 57.63 GB | Loss: 11.103
Step #1 | Batch Size: (64, 2048) | Time: 3.042s | Reserved Memory: 58.70 GB | Loss: 9.091
Step #2 | Batch Size: (64, 2048) | Time: 3.002s | Reserved Memory: 58.70 GB | Loss: 7.609

Cut Cross Entropy (Kahan Sum)
Step #0 | Batch Size: (64, 2048) | Time: 38.861s | Reserved Memory: 57.63 GB | Loss: 11.103
Step #1 | Batch Size: (64, 2048) | Time: 3.694s | Reserved Memory: 58.70 GB | Loss: 9.097
Step #2 | Batch Size: (64, 2048) | Time: 3.618s | Reserved Memory: 58.70 GB | Loss: 7.618

Regular Cross Entropy
Step #0 | Batch Size: (64, 2048) | Time: 55.47s | Reserved Memory: 73.76 GB | Loss: 11.10
Step #1 | Batch Size: (64, 2048) | Time: 2.431s | Reserved Memory: 73.77 GB | Loss: 9.097
Step #2 | Batch Size: (64, 2048) | Time: 2.371s | Reserved Memory: 73.77 GB | Loss: 7.616
```

Apple's implementation has a few good options to improve numerical stability (which I want to prioritize for similar convergence). For FP32 CCE vs Regular, memory drops by 23% (16GB) while step time increases by 41%. The experiment on my last blog used the default CCE implementation and it did not converge as well (I expect due to numerical instability). 

These are results with CCE vs baseline:
![Cut Cross Entropy results vs. my baseline](/Users/omkaarwork/Desktop/projects/personal-website/public/11-1b-model-p2/cce-run.png)

Verdict: I am close to thinking Chunked Cross Entropy is a scam. I will not be introducing this into my training regime. No amount of memory saving is worth losing training convergence.

### 8-bit AdamW
I already alluded to this in the last blog. I had a blunder in my experiment, so I re-did it and these are the results:
![Adam 8-bit vs 32-bit](/Users/omkaarwork/Desktop/projects/personal-website/public/11-1b-model-p2/adam-8vs32.png)

It seems to converge similarly which is very cool given it saves us a good chunk of memory. The MFU drop is on the order of 0.1% and so, is negligible. Memory needed goes down because of quantized weights but MFU takes a hit because of the de/quantization overhead before each computation.

Verdict: If I need more memory for other parts in the step, I introduce AdamW8bit. Even through empirically it works, I want to use the FP32 version as it is not lossy (compared to nf4) and has a slightly higher MFU.

### FP8 Math
H100 (or Hopper generally) supports FP8 (E4M3 & E5M2) natively. While we will keep the model in FP32, fp8 math should help us get much faster throughput, although de/quantizing will reduce it's effectiveness. 

For now, let's only touch the linear layers from our model.

#### Nvidia's Transformer Engine
First things first, I love the intuitiveness of [TE API](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb) to do FP8 linear ops. However, TE gives me same compute time but higher memory reserved:
```
Step #2 | Input Size: [4, 2048] | Time: 0.151s | Reserved Memory: 37.98 GB | Loss: 10.239
Step #3 | Input Size: [4, 2048] | Time: 0.141s | Reserved Memory: 42.28 GB | Loss: 9.975
Step #4 | Input Size: [4, 2048] | Time: 0.141s | Reserved Memory: 42.28 GB | Loss: 10.071
```

This is the recipe I used:
```
fp8_recipe = DelayedScaling(
    margin=1,
    interval=1,
    fp8_format=Format.HYBRID,
    amax_history_len=16,
)
```

I played around with the recipe but the performance numbers stayed the same... not useful.

#### Torch AO

TorchAO is interesting, the first time I tried it I must have made a mistake in it's implementation, it gave me significantly worse numbers. So, I dove deeper into the docs and found an excellent function for fp8 which I previously overlooked. I like that it works with autocast and compile... makes life easy.

All I had to do was add this after initializing the model:
```
ao_config = Float8LinearConfig.from_recipe_name("tensorwise")
convert_to_float8_training(model.layers, config=ao_config)
```

On [64, 2048] token inputs, with the `tensorwise` recipe, step time reduced by 10% while memory dropped by 20%!!! `rowwise` recipe might be more numerically stable, on my perf bench it was 2% slower than baseline but with the 20% drop. 

Let's train it for 1k steps and see their performance:
![FP8 Linear Graph vs FP32](/Users/omkaarwork/Desktop/projects/personal-website/public/11-1b-model-p2/fp8-linear.png)

The convergence for rowwise might be better but tensorwise works as well while being faster. 

#### FP8 Flash Attention
fp8 backward pass does not exist (only bf16/fp16 allowed), so I end up using SDPA. FWIW, I think FA3 was useful a few months back, torch team likely improved torch compile for Hopper.
 
### Overlapped DDP communication
DDP comms minning

### Compressed gradients in All Reduce
DDP comms minning

## Training convergence

I was tempted to use Muon in our last run but a good, simple baseline was the priority. In this post, I want to reason through different hyperparams and try out Muon. I only have a limited compute budget, so I refer to other works to shortcut some of this.

### Learning rate

What Karpathy did in [Nanochat](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L213) was very interesting. He assigned different learning rates for embedding, LM head and other matrices.

We already have a baseline with 3e-4 (a.k.a Karpathy Constant).

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
- Mix Attention - https://www.databricks.com/blog/mixattention

Help me think through all of this by looking stuff up. I need everything to be backed by science and industry best practices.



checkpointing - i want to checkpoint the model locally every 100 steps and every 1000 steps to a registry. 

logging metrics - wandb for logging every 10 steps (maybe 50?). I want to setup 


I will need to implement fine-grained data-mixing for this. I have a 20B token budget. 90% of that (or 18B tokens) will be 2048 token seq len, let's talk about that first. To have excellent english understanding, I will assign 50% of data to be from huggingface's filtered [fineweb-edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-10BT/train?p=2&row=290), 20% will be HF's [finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki), 20% code from [Star coder](https://huggingface.co/bigcode/starcoder) and the last 10% will be from [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math).