---
title: "LLM from Scratch"
date: "2026-09-25"
category: ""
description: ""
image: "/images/paste-1790510229659-609ac43580ceb19b.png"
---

# LLM from Scratch

This document follows *Build a Large Language Model from Scratch* by Sebastian Raschka. The goal is to develop a deeper understanding of how large language models process text, construct representations, perform inference, and are trained.

Rather than treating an LLM as a black box, the implementation is developed incrementally, starting from tokenization and ending with the mechanisms that allow the model to learn contextual representations.

## 1. Working with Textual Data

The first step in preparing text for an LLM is tokenization. Raw text cannot be directly processed by a neural network, so it must first be converted into a sequence of discrete tokens. A token can represent an individual word, part of a word, or a special character such as punctuation.

The text used for the initial experiments is *The Verdict*, a short story by Edith Wharton that has been released into the public domain. The text is available through Wikisource and was copied into a local text file named the-verdict.txt.

A simple tokenizer can be implemented by splitting the text using regular expressions and assigning each resulting token an integer identifier.

```python
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [x for x in result if x.strip() != '']
        encoded_list = [
            self.vocab[token] if token in self.vocab
            else self.vocab['<|UNK|>']
            for token in result
        ]
        return encoded_list

    def decode(self, token_ids):
        text = " ".join([self.inverse_vocab[i] for i in token_ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)
        return text
```

This tokenizer illustrates the basic relationship between text and the numerical representation consumed by the model. However, it also introduces an important limitation: words that were not present in the vocabulary cannot be represented directly.

## 2. Byte Pair Encoding

A naive tokenizer can handle unknown words by replacing them with a special <|UNK|> token. However, this loses useful information. For example, if the vocabulary does not contain the words "lower" and "lowest" , both could be mapped to the same <|UNK|> token even though they share meaningful structure.

Byte Pair Encoding (BPE) addresses this problem by representing words using smaller subword units. The tokenizer begins with individual characters and repeatedly merges the most frequent adjacent pair. Frequently occurring sequences eventually become single tokens, while rare words can still be represented using combinations of smaller tokens.

Consider the following small corpus:

```text
low
lower
lowest
low
```

Initially, each word is split into individual characters. We can mark the end of each word with a special symbol such as </w>`:

```text
l o w </w>
l o w e r </w>
l o w e s t </w>
l o w </w>
```

The tokenizer now counts how frequently adjacent pairs occur across the entire corpus. For example, the pair (l, o) appears in all four words:

```text
(l, o) = 4
```

The pair (o, w) also appears four times:

```text
(o, w) = 4
```

The pair (w, </w>) appears twice because "low" occurs twice:

```text
(w, </w>) = 2
```

The pair (w, e) appears twice because both "lower" and "lowest" contain "we"`:

```text
(w, e) = 2
```

Suppose BPE selects (l, o) as the most frequent pair. The pair is merged into a new token:

```text
l + o -> lo
```

The corpus is now represented as:

```text
lo w </w>
lo w e r </w>
lo w e s t </w>
lo w </w>
```

The pair frequencies are then calculated again. The pair (lo, w) now occurs four times, so it can be merged:

```text
lo + w -> low
```

The corpus becomes:

```text
low </w>
low e r </w>
low e s t </w>
low </w>
```

At this point, "low" has become a learned subword token because it occurs frequently throughout the corpus.

The next merge might be:

```text
low + </w> -> low</w>
```

This would represent the complete word "low" as a single token. However, the words "lower" and "lowest" can still be represented using the previously learned "low" token:

```text
lower  -> low + e + r
lowest -> low + e + s + t
```

If the corpus contains "lower" frequently enough, BPE may eventually learn another merge:

```text
low + e -> lowe
```

followed by:

```text
lowe + r -> lower
```

The resulting vocabulary therefore contains a mixture of complete words and subword units. Common patterns can become large tokens, while uncommon words can be decomposed into smaller tokens.

This is the main advantage of BPE. The tokenizer does not need to store every possible word in its vocabulary. Instead, it learns reusable pieces of words. A word that was never explicitly encountered during training can still be represented if its constituent subwords or characters are present in the vocabulary.

For example, after learning the appropriate merge rules, the tokenizer might represent:

```text
low      -> low
lower    -> lower
lowest   -> low + est
```

The exact resulting tokens depend on the training corpus and the number of merges performed. The important idea is that BPE builds its vocabulary incrementally by repeatedly replacing frequent adjacent token pairs with a new token.

## 3. Preparing the Training Data Using Sliding Windows

Once the text has been tokenized, the next step is to construct the training examples used by the language model. Autoregressive language models are trained to predict the next token given the tokens that came before it.

For a sequence such as:

```text
The cat sat on the mat
```

the model can be trained using input-target pairs such as:

```text
Input:  The
Target: cat

Input:  The cat
Target: sat

Input:  The cat sat
Target: on
```

In practice, rather than generating a separate training example for every position, a fixed-size context window is moved across the tokenized text. The stride determines how far the window moves after each training example.

```python
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, window, stride):
        self.text = text
        self.tokenizer = tokenizer
        self.window = window
        self.stride = stride
        self.x = []
        self.y = []

        self.data = self.tokenizer.encode(
            self.text,
            allow_special={"<|endoftext|>"}
        )

        for i in range(
            0,
            len(self.data) - self.window,
            self.stride
        ):
            input_chunk = self.data[i:i+self.window]
            target_chunk = self.data[i+1:i+self.window+1]

            self.x.append(input_chunk)
            self.y.append(target_chunk)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_data_loader(
    text,
    batch_size=2,
    window=256,
    stride=128,
    drop_last=True,
    shuffle=True
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(
        text,
        tokenizer,
        window,
        stride
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0
    )

    return data_loader
```

The resulting input and target sequences differ by one token. This shift is what creates the next-token prediction objective that forms the basis of autoregressive language modeling.

## 4. Creating Token Embeddings

Neural networks operate on numerical vectors rather than discrete token identifiers. Although a token can initially be represented by an integer such as 42 , the integer itself does not contain any meaningful notion of similarity. For example, the model should not interpret token 43 as being inherently more similar to token 42 than token 900 .

Embedding layers solve this problem by mapping each token identifier to a continuous-valued vector.

An embedding layer can be viewed as a learnable matrix in which each row corresponds to a token's vector representation. During training, these vectors are updated through backpropagation, allowing the model to gradually learn representations that are useful for the language modeling task.

Conceptually, an embedding layer performs a transformation similar to a linear operation, but it is implemented as a lookup operation because only the rows corresponding to the input token IDs are required. This makes embedding layers computationally efficient for large vocabularies.

## 5. Self-Attention Mechanism

A token embedding by itself does not contain information about the other tokens in the sequence. The meaning of a token often depends heavily on its surrounding context. Self-attention provides a mechanism through which each token can incorporate information from other tokens in the sequence.

A useful way to think about the result of attention is as a **context vector**. A context vector has the same dimensionality as the representation being processed and contains a weighted combination of information from the tokens that the current token attends to. In other words, it is a representation of the current token that has been modified according to its surrounding context.

The self-attention mechanism used in the Transformer architecture, including GPT models, is commonly referred to as **scaled dot-product attention**. Instead of directly using the token embeddings to determine the attention relationships, the model projects every input into three different representations: queries, keys, and values.

These representations are produced using three trainable weight matrices:

```text
Q = XWq
K = XWk
V = XWv
```

The query and key representations determine how strongly tokens should attend to one another, while the value representations contain the information that is ultimately combined to produce the context vectors.

The attention scores are obtained through the dot product between queries and keys:

```text
Attention Scores = QKᵀ
```

A larger dot product indicates greater compatibility between a query and a key. The scores are then scaled by the square root of the key dimensionality before applying softmax. This scaling prevents the dot products from becoming excessively large as the dimensionality increases.

## 6. Causal Attention

For an autoregressive language model, ordinary self-attention introduces a problem. If every token is allowed to attend to every other token, a token could access information from future positions in the sequence.

Consider the sequence:

```text
The cat sat on the mat
```

When predicting the token after "The cat sat on the" , the model should only have access to the tokens that occur before or at the current position. It must not be allowed to inspect "mat" because "mat" is precisely the information the model is supposed to predict.

This restriction is implemented using **causal attention**, also called masked self-attention. The attention matrix is modified so that each position can attend only to itself and the positions before it.

For example, a sequence of four tokens produces a mask with the following structure:

```text
1  0  0  0
1  1  0  0
1  1  1  0
1  1  1  1
```

The upper triangular portion represents future tokens and must therefore be excluded from the attention calculation.

One important detail is that the forbidden positions should not simply be assigned a score of zero. The attention scores are converted into probabilities using softmax, and zero is still a valid numerical score. For example:

```text
scores = [2, 1, 3, 0]
```

produces non-zero probability for the final element. A zero score therefore does not mean "do not attend to this token."

Instead, the forbidden positions are assigned negative infinity:

```text
scores = [2, 1, 3, -∞]
```

When softmax is applied, the exponential of negative infinity becomes zero:

```text
e^(-∞) = 0
```

Consequently, the forbidden position receives exactly zero attention probability. This is why causal masks conventionally use -torch.inf rather than zero.

The order of operations is also important. The mask must be applied **before** softmax:

```text
QKᵀ
  ↓
Scaling
  ↓
Causal Mask
  ↓
Softmax
  ↓
Attention Weights
  ↓
Weighted Sum of V
```

Applying the mask after softmax would require the probabilities to be renormalized because simply setting some probabilities to zero would cause their sum to become less than one.

The following implementation applies the causal mask directly to the attention scores before the softmax operation.

```python
import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias):
        super().__init__()

        self.Wq = torch.nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.Wk = torch.nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.Wv = torch.nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attn_scores = torch.matmul(Q, K.T)

        context_length = attn_scores.shape[0]

        mask_simple = torch.triu(
            torch.ones(
                context_length,
                context_length
            ),
            diagonal=1
        )

        attn_scores = attn_scores.masked_fill(
            mask_simple.bool(),
            -torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores / K.shape[-1]**0.5,
            dim=1
        )

        context_vector = attn_weights @ V

        return context_vector
```

The causal mask does not modify the queries, keys, or values themselves. Instead, it determines which key positions are available to each query when the attention probabilities are calculated. As a result, the context vector for a token can incorporate information from its previous context without allowing the model to access future tokens.

The trainable matrices Wq , Wk , and Wv are what make this attention mechanism learnable. During training, their parameters are updated through backpropagation. The model therefore learns projections that allow the attention mechanism to construct useful contextual representations for the language modeling objective.


## 7. Multihead Attention

The self-attention mechanism described above uses a single set of query, key, and value projections. The Transformer architecture extends this idea through **multi-head attention**, in which the representation is divided into several smaller attention mechanisms called heads.

The motivation is that different attention heads can learn different relationships between tokens. One head may learn to focus on nearby words, while another may learn relationships between words that are farther apart in the sequence. The model does not explicitly assign these roles to the heads. They emerge from the learned parameters during training.

Suppose the output dimension of the attention layer is d_out = 8 and the model uses two attention heads. The representation is divided equally between the heads:

```text
d_out = 8
num_heads = 2

head_dim = d_out / num_heads
         = 8 / 2
         = 4
```

Each attention head therefore operates on a four-dimensional representation. The two heads are later combined to reconstruct the original eight-dimensional representation.

The implementation begins by projecting the input into queries, keys, and values:

```python
keys = self.W_key(x)
queries = self.W_query(x)
values = self.W_value(x)
```

If the input has the shape

```text
[B, T, d_in]
```

where B is the batch size and T is the number of tokens, the three projections have the shape

```text
Q, K, V : [B, T, d_out]
```

The output dimension is then divided into separate attention heads. This is performed using view`:

```python
keys = keys.view(
    b,
    num_tokens,
    self.num_heads,
    self.head_dim
)

queries = queries.view(
    b,
    num_tokens,
    self.num_heads,
    self.head_dim
)

values = values.view(
    b,
    num_tokens,
    self.num_heads,
    self.head_dim
)
```

For example, a tensor with the shape

```text
[B, 6, 8]
```

becomes

```text
[B, 6, 2, 4]
```

when using two heads. The eight-dimensional representation has therefore been reorganized into two four-dimensional representations.

The next operation is a transpose:

```python
keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)
```

This changes the shape from

```text
[B, T, H, D]
```

to

```text
[B, H, T, D]
```

where H is the number of attention heads and D is the dimensionality of each head. Placing the head dimension before the token dimension allows the matrix operations that follow to perform attention independently for every head.

The attention scores are calculated using the dot product between queries and keys:

```python
attn_scores = queries @ keys.transpose(2, 3)
```

The shapes involved are

```text
queries : [B, H, T, D]
keysᵀ   : [B, H, D, T]

result  : [B, H, T, T]
```

Thus, every attention head produces its own T × T attention matrix. Each row describes how strongly a particular token attends to every other token.

Because this implementation is intended for an autoregressive language model, causal masking is still required. Future tokens must not be visible when predicting the next token. The mask is therefore applied before the softmax operation:

```python
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

The resulting -∞ values become zero after softmax, preventing the corresponding future tokens from contributing to the context vector.

The scores are then scaled according to the dimensionality of each individual head:

```python
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5,
    dim=-1
)
```

The use of keys.shape[-1] is important because attention is calculated separately within each head. If d_out = 8 and there are two heads, each head has a dimensionality of four, so the scaling factor is sqrt(4) rather than sqrt(8)`.

The attention weights are then multiplied by the value representations:

```python
context_vec = attn_weights @ values
```

The resulting tensor has the shape

```text
[B, H, T, D]
```

Each head has therefore produced its own contextual representation of every token.

The individual heads are then combined. First, the token and head dimensions are rearranged:

```python
context_vec = context_vec.transpose(1, 2)
```

which changes the shape to

```text
[B, T, H, D]
```

The head and head-dimension axes can then be flattened:

```python
context_vec = context_vec.contiguous().view(
    b,
    num_tokens,
    self.d_out
)
```

This reconstructs the original output dimension:

```text
[B, T, H, D]
        ↓
[B, T, H × D]
        ↓
[B, T, d_out]
```

The outputs of the individual heads have effectively been concatenated into a single representation.

A final learned linear projection is then applied:

```python
context_vec = self.out_proj(context_vec)
```

The purpose of this projection is to allow the model to mix information from the different attention heads after their outputs have been combined. The complete flow can therefore be summarized as:

```text
Input
  │
  ├── Wq ──→ Queries ──┐
  ├── Wk ──→ Keys ─────┼──→ Split into heads
  └── Wv ──→ Values ───┘
                         │
                         ↓
                  Attention per head
                         │
                         ↓
                   Causal masking
                         │
                         ↓
                      Softmax
                         │
                         ↓
                    Weighted V
                         │
                         ↓
                 Concatenate heads
                         │
                         ↓
                     Wₒ projection
                         │
                         ↓
                    Output
```

The main difference between single-head and multi-head attention is therefore not a fundamentally different attention operation. Each head performs the same scaled dot-product attention described previously, but on a different learned subspace of the representation. The outputs are then combined to form the final contextual representation.

This gives the Transformer multiple independent attention mechanisms that can learn different relationships within the same sequence.

```python
from torch import nn
import torch
class MultiHeadAttention(nn.Module):
 def __init__(self, d_in, d_out,
 context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0), \
    "d_out must be divisible by num_heads"
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
    "mask",
    torch.triu(torch.ones(context_length, context_length),
    diagonal=1)
    )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x) 
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
        b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec
```

## 8. Putting the Transformer together

After defining token embeddings, positional embeddings, multi-head self-attention, normalization, and the feed-forward network, we can combine these components into the overall GPT architecture. The GPTModel class represents the complete language model that transforms a sequence of token IDs into a probability distribution over the vocabulary for every position in the sequence.

The model begins with two embedding layers. The token embedding layer converts each token ID into a dense vector of size emb_dim. However, token embeddings alone do not tell the model where a token occurs in the sequence. A second embedding layer, the positional embedding layer, therefore assigns a learned vector to each position in the context window. The two embeddings are added together:

Token representation = Token embedding + Positional embedding

This gives every token representation both semantic information about the token itself and information about its position in the sequence.

In the implementation, the positional indices are generated with:

```python
torch.arange(seq_len, device=in_idx.device)
```

If the input contains a sequence of length 5, this produces the positions 0 through 4. These positions are passed through the positional embedding layer, producing one learned positional vector for each position. The resulting positional embeddings have the same dimensionality as the token embeddings, which makes element-wise addition possible.

After combining token and positional information, dropout is applied:

```python
x = self.dropout(x)
```

Dropout randomly sets some elements of the representation to zero during training. This prevents the model from becoming overly dependent on particular activation values and acts as a form of regularization. During evaluation, dropout is automatically disabled.

The resulting representation is then passed through a sequence of Transformer blocks:

```python
x = self.trf_blocks(x)
```

The number of blocks is determined by cfg["n_layers"]:

```python
self.trf_blocks = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
)
```

This creates multiple Transformer blocks and places them sequentially. Each block processes the representation using self-attention and a feed-forward network. Stacking these blocks allows the model to progressively construct more sophisticated representations. Earlier layers can capture relatively simple relationships, while later layers can combine information from previous transformations into increasingly complex representations.

## Layer Normalization

After the Transformer blocks, the representation passes through a final normalization layer:

```python
x = self.norm(x)
```

The custom NormalizationLayer implements Layer Normalization. For each token representation, it calculates the mean and variance across the embedding dimension:

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
```

The representation is then standardized by subtracting its mean and dividing by its standard deviation:

```text
normalized = (x - mean) / sqrt(variance + epsilon)
```

The purpose is not to force the model to produce a perfect normal distribution. Instead, LayerNorm keeps the numerical scale of the activations under control. This is useful because a Transformer contains many successive transformations and residual connections, and uncontrolled activation magnitudes can make optimization more difficult.

However, the model is not permanently forced to use normalized values. The normalization layer contains two learnable parameters:

```python
self.scale = nn.Parameter(torch.ones(emb_dim))
self.shift = nn.Parameter(torch.zeros(emb_dim))
```

These are the learnable scale and shift parameters. After normalization, the layer computes:

```text
output = scale * normalized + shift
```

Initially, scale is one and shift is zero, so the layer simply performs the normalization. During training, gradient descent can modify both parameters. This allows the model to learn whatever scale and offset are useful for the task while still benefiting from the stabilizing effect of normalization.

This is an important property of LayerNorm. The normalization provides a controlled numerical environment, while the learnable scale and shift prevent that normalization from unnecessarily restricting what the network can represent.

## From Representations to Vocabulary Predictions

After normalization, the resulting representation is passed to the output head:

```python
logits = self.out_head(x)
```

The output head is a linear transformation:

```python
self.out_head = nn.Linear(
    cfg["emb_dim"],
    cfg["vocab_size"],
    bias=False
)
```

This changes the dimensionality from emb_dim to vocab_size.

For example, suppose the embedding dimension is 768 and the vocabulary contains 50,000 tokens. Every token representation with 768 values is transformed into 50,000 values.

Each of these 50,000 values is a logit representing the model's unnormalized score for one vocabulary token.

If the input has the shape:

```text
[batch_size, sequence_length]
```

the embedding layers produce:

```text
[batch_size, sequence_length, embedding_dimension]
```

The Transformer blocks preserve this overall shape. The final output head then produces:

```text
[batch_size, sequence_length, vocabulary_size]
```

Therefore, for every position in every sequence, the model produces one logit for every possible vocabulary token.

For example, consider the input:

```text
The cat sat on
```

The representation at the final position is transformed into a vector containing a score for every possible next token. Depending on what the model learned during training, tokens such as "the", "mat", or "floor" might receive relatively high scores.

The logits are not probabilities yet. During training, they are passed to a cross-entropy loss, which compares the predicted scores against the correct next token. During inference, the logits can be passed through softmax to convert them into probabilities.

## The Feed-Forward Network

Another important component inside every Transformer block is the feed-forward network:

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

The feed-forward network first expands the embedding dimension by a factor of four. If the embedding dimension is 768, the first linear layer transforms each token representation from 768 dimensions into 3072 dimensions:

```text
768 → 3072
```

The GELU activation is then applied. This introduces the nonlinearity required for the network to learn transformations that cannot be represented by linear operations alone.

Finally, another linear layer projects the representation back to its original dimensionality:

```text
3072 → 768
```

The expansion followed by contraction gives the network a larger intermediate space in which it can transform the representation.

Conceptually, self-attention and the feed-forward network perform different jobs. Self-attention allows each token to incorporate information from other relevant tokens in the sequence, while the feed-forward network applies a learned nonlinear transformation to each token representation independently.

The overall flow of the model can therefore be summarized as:

```text
Token IDs
    ↓
Token Embeddings + Positional Embeddings
    ↓
Dropout
    ↓
Transformer Blocks
    ↓
Layer Normalization
    ↓
Linear Output Head
    ↓
Logits for Every Vocabulary Token
```

This is the fundamental computation performed by a GPT-style language model. The Transformer blocks repeatedly refine the token representations, LayerNorm keeps their numerical behavior under control, and the final linear layer converts those representations into scores over the entire vocabulary.

During training, the predicted scores are compared with the correct next tokens. The resulting loss is then propagated backward through the entire architecture, allowing the model to update the parameters of the embeddings, attention layers, feed-forward networks, normalization layers, and output head.
