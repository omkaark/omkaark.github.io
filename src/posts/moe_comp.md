# Foray into MoE compression

I think the Qwen3-Coder-235B-45B model is decent for short-to-mid-sized tasks like code completion and making small edits. As code agents, the model still suffers from the same probelms as older Claude sonnet and Codex versions.

I am wrapping my head with where we will be 4-8 months down the line. I think Open Source models are a great way for new labs to enter the market but whether they continue their strategy or continue to be generous is uncertain.

That being said, we still have 2-3 great OS models coming up. I see model releases a bit differently than other people. To me, a pre-train + post-train release (base + instruct) is >>> valuable than just a post-train release. A pre-train is expensive ($X+ Ms) and labs only conduct pre-training runs when there are drastic changes to architecture (usually the next generation). Any edit to the model behaviour other than architectural can very likely be done though mid-training (continual pre-train on longer context data / instruct-tuning data / data that aids upcoming RL) + post-training (preference-tuning / RL for thinking / specialized capabilities). mid- + post-training is usually cheaper (questionable in the RL paradigm though) as we train on OOMs less data.

My general prediction is more and more enterprises will start fine-tuning and RL'ing on proprietary data and reduce reliance on labs. Labs train for general use-cases but I only care about performance on my relevant evals not some random, hill-climbed eval labs use for marketing.

A boon with models trained for your own evals is you don't need massive models anymore, a good SFT + RL run on artisanal datasets can do wonders. A very simple example of this is [my SFT run](https://omkaark.com/posts/tinker-fim.html) to take a small qwen model from 0% to 28% on my code FIM evals with 10M tokens.

MoEs are hard to train but supposedly make inference much better. They have the knowledge of a large model with the compute/token of a small model (commonly referred to as sparsity). Till we come up with bullet-proof load-balancers for MoEs, they suffer from one problem: high expert redundancy. Many experts end up having common knowledge so the actual approximate rank of the expert layer is much lower.

Today, I want to exploit that and try out a bunch of papers to reduce MoE model size and evaluate them on my code FIM evals. Techniques are usually judged on their ability to keep perplexity low but I will be using Qwen3-Coder-30B-A3B on my FIM evals as we only care about downstream usecases. These learnings are more broadly applicable to wider companies.

I will go through different kinds of literature and I plan to implement a few of them that I intuitively think work the best.

## Quick MoE primer

If you already know how MoEs work, skip to the next section.

Most people are familiar with dense LLMs which have either one FFN or SwiGLU per decoder block. 

MoEs simply have multiple smaller FFN's in parallel per decoder block. The catch and what makes it really efficient is that only n/N experts per layer are "active" or forward passed during inference per token. 

Which n experts get activated is based on the load balancer which is another trained linear layer. Making expert layers less redundant is a function of getting this load balancer trained right (open research question).

## Types of Compression

I have exciting literature to cover today, at a high level, these are:
- Pruning - Deleting weights
- Merging - Merging experts
- Re-modelling experts

Before we get into that, I have an int4-quantized Qwen3-Coder-30B-A3B model which achieves 67% compile success and 40% test passes on my code FIM benchmark. In summary, the eval measures the model's ability to generate fill-in-middle snippets that can compile and pass unit tests. It's a simple but effective eval. I want to retain that goal as much as possible.

### Pruning

Pruning itself has two main ways to do it: structured (delete whole experts) and unstructured (delete individual weights). While there are many papers like [STUN](https://arxiv.org/abs/2409.06211) and [ MoE-Pruner](https://arxiv.org/abs/2410.12013), I will implement this paper by [Cerebras](https://arxiv.org/pdf/2510.13999) as I see it being used quite a lot.

