# How to Train an LLM: Part 5
Welcome back to the continuation of [Part 1](https://omkaark.com/posts/llm-1b-1.html), [Part 2](https://omkaark.com/posts/llm-1b-2.html), [Part 3](https://omkaark.com/posts/llm-1b-3.html), and [Part 4](https://omkaark.com/posts/llm-1b-4.html) of the saga.

![](https://raw.githubusercontent.com/omkaark/omkaark.github.io/refs/heads/main/public/13-1b-model-p5/headline.png?raw=true)

In the last worklog, I started my training run for the final model. Well, the models have been baked in. I actually started two runs, one with the standard llama + QKNorm architecture and the other with llama + QKNorm + MixAttention (cross-doc masking in both).

In this worklog, I'll showcase the final base model, run my eval on it, and give a retrospective.

Cover:
- Model
- Eval model
- Pre-training doesn't make sense for me. Labs are training 1:1000 param:data, I might have the data sorted but no chance for that much compute. Even though pre-training lets me choose an architecture better suited for my inference needs, integrating it in downstream systems will be a pain. All this pain can be skipped for a tiny cost, just use a post-trained OS model.
- This is a learning that a bunch of companies in the space are realizing. I still think I have a good idea of a pre-training solid code FIM model, but that has to come after revenue.

## Introduction

## Conclusion


Omkaar Kamath