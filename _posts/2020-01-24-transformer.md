---
layout: post
title: "Transformer" (Draft)
output: pdf_document
usemathjax: true
---


The aim of this post is to describe the original Transformer architecture in details, and then provide a flavour of the subsequent transformations (:P) of the model. This post could be math heavy, so a friendlier version of this post will be attempted in an upcoming post.

# Motivation

The Transformer architecture aims to address the problem of [Sequence Transduction](https://arxiv.org/abs/1211.3711), which involves transformation of input sequences to output sequences e.g. machine translation, speech recognition, text-to-speech. Transformers map sequences of input vectors $(x_1, ..., x_n)$ to sequences of output vectors $(y_1, ... , y_n)$ of the same length.  

Transformers can be any sequence based model that primarily uses *self-attention* to propagate information along the time dimension. 

# Self-attention

Self-attention allows a network to directly extract and use information from arbitrarily large contexts (unlike RNNs which would need to pass it through intermediate recurrent connections).

Self-attention can manifest in multiple forms, namely *Encoder-Decoder attention*, *Causal self-attention* and *Bi-directional self-attention*.

## Encoder-Decoder attention

One sentence (decoder) looks at another sentence (encoder) e.g. in a Machine Translation model where a French sentence looks at an English sentence.

## Causal (or masked) self-attention

In a given sentence, words only look at previous words e.g. in autoregressive text generation (summarization). In this case, 
- the model has access to information about inputs upto and including the current input, but no information regarding the inputs beyond.
- the computation performed for each item is independent of other computations. This means we can parallelize the forward inference and training of these models.

## Bi-directional self-attention

In one sentence, words look at both previous and upcoming words. This mechanism is used in models like BERT and T5 (these will be discussed later).


# Causal self-attention in detail

Attention, at its core, 

Simple attention calculation


$$ 
similarity\_score (x_i, x_j) = x_i \cdot x_j 
$$

This dot product results in a scalar value, $ -\infty < similarity\_score (x_i, x_j) < \infty$, so a softmax normalization can be applied to map the values to $[0,1]$ and to ensure that it sums to $1$ over the entire sequence.

$$ w_{i, j} = softmax(similarity\_score(x_i, x_j)) \quad \forall j \leq i\\
 = \frac{exp(similarity\_score(x_i, x_j))}{\sum_{k=1}^i exp(similarity\_score(x_i, x_j))} \quad \forall j \leq i
$$

$$
y_i = \sum_{j \leq i} w_{i,j} x_j
$$


## Query, Key and Value parameters

$$ q_i = W^Q x_i; \quad k_i = W^K x_i; \quad v_i = W^V x_i $$

$$ score(x_i, x_j) = q_i \cdot k_j $$

$$ y_i = \sum_{j \leq i} \alpha_{i, j} v_j $$

$$ score(x_i, x_j) = \frac{q_i \cdot k_j}{\sqrt{d_k}} $$


## Final formulation

$$ Q = W^Q X; \quad K = W^K X; \quad V = W^D X $$

$$ SelfAttention(Q, K, V) = \text{softmax} (\frac{Q K^T}{\sqrt{d_k}}) V $$




# Model Architecture

## Masking

In case of autoregressive models, masking makes the Transformer models causal. This is so that it does not know the upcoming value from the training data, but instead only care about the earlier positions in the output sequence. Masking should be done in the decoder.

## Positional Embedding

To ensure that Transformer cares about the order of words.

## Positional Encoding 

Encodings created through some funcitons that map the positions to some real valued vectors. A well chosen function could mean that the transformer is able to deal with sequence lengths longer that the ones it has seen during training.

# Knowledge Check

1. What is an autoregressive model?
2. Why do we need to normalize the comparison score between two word embeddings?
3. What is the difference between an encoding and an embedding?
4. Is multi-head attention > simple attention? Why?

# References

[1] Peter Bloem, Transformers From Scratch  
[2] Daniel Jurafsky & James Martin, Deep Learning Architectures for Sequence Processing