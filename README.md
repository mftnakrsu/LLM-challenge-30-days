# 🚀 LLM Challenge: 30 Days

## **Objective**
This repository is dedicated to a **30-day journey to mastering LLM (Large Language Models), RAG (Retrieval-Augmented Generation), AI Agents, and Generative AI** through structured learning and hands-on projects.


# Day 1 
### **Understanding Transformer-Based Language Models**

#### **1. Introduction to Transformers**  
Transformer models have transformed natural language processing (NLP) by introducing a parallelizable and highly efficient approach to text processing. Introduced in the groundbreaking paper *"Attention is All You Need"*, the Transformer architecture eliminates the sequential dependencies of recurrent neural networks (RNNs) and long short-term memory (LSTM) models, leading to significant improvements in training efficiency and model performance.

---

#### **2. Transformer Architecture**  
The Transformer consists of two main components:  

- **Encoder:** Processes input text by capturing relationships between words using self-attention and feed-forward neural networks.
- **Decoder:** Generates output predictions by attending to both previously generated tokens and encoder outputs.

This structure is particularly effective in tasks such as **machine translation**, **text summarization**, and **question answering**.

---

#### **3. Self-Attention Mechanism**  
A key feature of the Transformer model is **self-attention**, which enables the model to weigh the importance of different words within a sentence. Unlike traditional sequential models, self-attention allows the model to process all words simultaneously, capturing long-range dependencies and contextual relationships efficiently.  

- **Masked Self-Attention:** Used in generative models like GPT to ensure causality by restricting attention to previous tokens only.

This mechanism allows the model to **prioritize words that contribute most to meaning**, significantly improving contextual understanding.

---

#### **4. BERT: A Representation Model**  
BERT (*Bidirectional Encoder Representations from Transformers*) is an encoder-based model designed for **language understanding tasks**.  

##### **Key Features of BERT:**  
- **Bidirectional Attention:** Considers both past and future words in a sentence, leading to deeper contextual word embeddings.
- **Masked Language Modeling (MLM):** Trains the model by randomly masking words in a sentence and predicting them.
- **Fine-Tuning for Specific Tasks:** After pre-training on large datasets, BERT can be fine-tuned for **text classification, named entity recognition (NER), question answering**, and **sentiment analysis**.

BERT is highly effective for extracting meaning from text and understanding word relationships.

---

#### **5. GPT: A Generative Model**  
GPT (*Generative Pre-trained Transformer*) is a **decoder-only** model optimized for text generation.  

##### **Key Features of GPT:**  
- **Autoregressive Learning:** Predicts the next word in a sequence based on previous words.
- **Masked Self-Attention:** Ensures that the model does not "see" future tokens when generating text.
- **Fluent and Coherent Output:** Excels in applications such as **chatbots, text completion, and creative writing**.

While BERT is designed for understanding text, GPT is more suited for **generating human-like responses**.

---

#### **6. Comparison: BERT vs. GPT**  

| Feature | BERT | GPT |
|---------|------|-----|
| Architecture | Encoder-based | Decoder-based |
| Attention | Bidirectional | Unidirectional |
| Training Objective | Masked Language Modeling (MLM) | Autoregressive Text Generation |
| Use Cases | Text classification, NER, question answering | Chatbots, text generation, creative writing |

BERT is better for **extracting information**, while GPT is better for **generating new content**.

---

#### **7. Transformer-Based Language Models: Open-Source vs. Proprietary**  
The NLP space is now divided between **proprietary** and **open-source** language models.  

##### **Proprietary Models (Closed-Source):**  
- **GPT-4** *(OpenAI)*  
- **Gemini** *(Google DeepMind)*  
- **Claude 2** *(Anthropic)*  

These models are state-of-the-art but **restricted in access and control**.

##### **Open-Source Models:**  
- **Llama 2** *(Meta)*  
- **Falcon** *(Technology Innovation Institute)*  
- **Mistral** *(Mistral AI)*  

Open-source models provide **flexibility, transparency, and customization**, making them suitable for research and enterprise applications.

---

#### **8. Conclusion: The Future of Transformers**  
Transformer models have revolutionized NLP by making **language understanding and generation more powerful than ever**. With their **self-attention mechanisms** and **scalability**, they are widely used in applications such as **machine translation, text summarization, search engines, and AI chatbots**.  

As **new models continue to emerge**, the distinction between **representation models (BERT)** and **generative models (GPT)** remains crucial for selecting the right tool for each NLP task.

# DAY 2

### **Tokenization and Representation**  
Tokenization is a critical step where input text is broken into smaller units, such as words, subwords, or characters. These tokens are mapped to numerical representations using an embedding matrix. The model's tokenizer maintains a predefined vocabulary, assigning each token a unique ID. The embeddings capture semantic relationships between words, which help the model understand context.

---

### **Transformer Processing – Parallelization vs. Sequential Generation**  
Unlike traditional RNNs, transformers process all input tokens simultaneously in parallel, leveraging self-attention mechanisms to weigh relationships between words. This parallelization allows for highly efficient training and inference. However, during text generation, transformers generate tokens sequentially, producing one token at a time while considering previously generated ones.

---

### **Decoding Strategies – Greedy Decoding vs. Temperature Sampling**  
The decoding process determines how the model selects the next token in a sequence. Two primary methods were discussed:  

- **Greedy Decoding (Temperature = 0):** At each step, the model picks the token with the highest probability. This results in deterministic outputs but may lack diversity.  
- **Temperature Sampling (Temperature > 0):** When temperature is greater than zero, the probability distribution is adjusted to introduce variability. A higher temperature makes the output more diverse and creative, while a lower temperature makes it more deterministic.  
- **Top-k and Top-p Sampling:** These techniques dynamically filter the token selection process by limiting the vocabulary to the most probable tokens (top-k) or adjusting the probability mass threshold (top-p or nucleus sampling).  

The choice of decoding strategy significantly impacts the fluency and creativity of generated text.

---

### **KV Caching – Optimizing Inference for Sequential Generation**  
During text generation, transformers generate tokens one by one in an autoregressive manner. To improve efficiency, **KV (Key-Value) Caching** is used to store previously computed hidden states (key-value pairs). Instead of recomputing the self-attention mechanism from scratch for each new token, the model reuses stored computations. This reduces redundant operations and speeds up inference, especially for long sequences.

---

### **Inference Pipeline and Cached Computation**  
When processing input prompts, LLMs follow a structured inference pipeline:
1. **Tokenization:** The prompt is split into tokens and mapped to numerical embeddings.
2. **Transformer Block Computation:** The model applies self-attention and feed-forward networks to process the input.
3. **Sequential Token Generation:** In autoregressive models, the next token is generated one by one based on previous outputs.
4. **KV Caching Implementation:** Previously computed attention keys and values are stored and reused to speed up processing.
5. **Decoding Strategy Application:** The model selects the most probable token based on the chosen decoding strategy (greedy, temperature-based, top-k, etc.).
6. **Final Output Assembly:** The generated tokens are combined to produce coherent text.

---

### **Application of LLMs in Text Generation**  
The practical applications of these techniques span various domains, including:
- Automated content generation (emails, reports, chat responses)
- Text summarization
- Language translation
- Conversational AI
- Code generation and completion  

# Transformer Concepts
## Key Components of Transformers

### Tokenization and Embeddings
Before passing input into a transformer model, text is tokenized and converted into numerical representations called **embeddings**. The tokenizer holds a vocabulary of tokens, mapping each to a unique token ID. The embedding layer then converts these IDs into high-dimensional vector representations.

Example of tokenization:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("The Shawshank Redemption")
print(tokens)  # ['the', 'shawshank', 'redemption']
```

### Self-Attention Mechanism
Self-attention allows the model to weigh the importance of different words in a sentence relative to each other. It computes three vectors:
- **Query (Q)**: Represents the current token
- **Key (K)**: Represents the context for attention
- **Value (V)**: Holds the information to be passed forward

The attention score is calculated using:
```
Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) V
```
where **d_k** is the dimension of key vectors, preventing overly large values.

### Feed-Forward Neural Networks (FFNN)
After self-attention, each token embedding passes through a fully connected feed-forward network (FFNN). The transformation can be described as:
```
FFN(x) = max(0, xW1 + b1) W2 + b2
```
where **W1, W2** are weight matrices, and **b1, b2** are biases.

### Multi-Head Attention
Instead of a single attention mechanism, transformers use multiple attention heads to capture different contextual meanings. The outputs of multiple attention heads are concatenated and linearly transformed.

## Decoding Strategies for Text Generation
When generating text, models use different decoding strategies:

### Greedy Decoding
Selects the token with the highest probability at each step.
```python
def greedy_decoding(model, input_ids):
    output = model.generate(input_ids, max_length=50)
    return output
```
### Temperature Sampling
Controls randomness in sampling, with higher values making the output more creative.
```
P(t) = exp(logit_t / temperature) / sum(exp(logit_i / temperature))
```
where **temperature > 1** increases randomness, and **temperature = 0** is deterministic (greedy decoding).

### Top-K and Top-P Sampling (Nucleus Sampling)
- **Top-K** restricts choices to the top K most probable tokens.
- **Top-P (Nucleus Sampling)** selects from the smallest set of tokens whose probabilities sum to a threshold p.

```python
def nucleus_sampling(model, input_ids, top_p=0.9):
    output = model.generate(input_ids, do_sample=True, top_p=top_p, max_length=50)
    return output
```

## KV (Key-Value) Caching for Efficient Decoding
When generating long sequences, transformers use **KV caching** to store previous key and value vectors, reducing redundant computation and speeding up inference.

## Example: Using a Transformer Model
Here’s an example using Hugging Face’s transformers library to generate text with GPT-2:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=50, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Coreference Resolution
Coreference resolution helps models identify when different words refer to the same entity in a text. For example:
"John went to the store. He bought milk."
The model should recognize that "He" refers to "John."


# DAY 3

### **Relevance Scoring and Combining Information in Self-Attention**

In self-attention mechanisms, **relevance scoring** determines how much focus a token should give to other tokens in a sequence. This is achieved using the **Scaled Dot-Product Attention** formula:

```
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
```

where:
- **Q (Query):** The token looking for relevant information.
- **K (Key):** Other tokens being compared.
- **V (Value):** The actual information retrieved.
- **d_k:** A scaling factor.

#### **Step 1: Relevance Scoring**
Each token computes its relevance to all other tokens using **dot-product similarity** between Q and K. Higher dot-product values indicate stronger relationships. The softmax function normalizes these scores.

##### **Example**
Consider the sentence:

```
The cat sat on the mat because it was tired.
```

To resolve **"it"**, the model needs to decide whether it refers to **"the cat"** or **"the mat"**. Using relevance scoring, the self-attention mechanism assigns **higher weights** to **"the cat"** based on context.

#### **Step 2: Combining Information**
Once scores are computed, they are used to weight the corresponding **V** values. The output is a weighted sum of all tokens.

##### **Python Example**
```python
import numpy as np

Q = np.array([[1, 0.5]])  # Query token
K = np.array([[1, 0.5], [0.3, 0.8]])  # Key tokens
V = np.array([[0.2, 0.7], [0.6, 0.1]])  # Value tokens

# Compute dot-product similarity
scores = np.dot(Q, K.T)

# Apply softmax to get attention weights
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

# Compute final output
output = np.dot(attention_weights, V)

print("Relevance Scores:\n", scores)
print("Attention Weights:\n", attention_weights)
print("Final Combined Representation:\n", output)
```

#### **Metaphor**
Imagine you are in a meeting, and multiple people are speaking. Your brain **scores** each speaker based on relevance—your boss’s words may have more weight than casual comments. You then **combine** this information, prioritizing important insights while still considering others.

This **dynamic weighting mechanism** is crucial for:
- **Long-range dependencies** (capturing relationships between distant words).
- **Coreference resolution** (linking pronouns to the correct entity).
- **Contextual understanding** (refining meaning based on the full sentence).
Here is a structured summary of the latest images focusing on key concepts relevant to Transformers and Large Language Models:

---

# **Advanced Attention Mechanisms in Transformers**

## **1. Self-Attention Mechanism Breakdown**
### **Query, Key, and Value Projections**
- Each input token is transformed into three distinct vectors:
  - **Query (Q):** Represents what the token is looking for in the sequence.
  - **Key (K):** Represents the content of each token in the sequence.
  - **Value (V):** Contains the actual information associated with each token.
- These projections are performed using learned weight matrices.

### **Computing Attention Scores**
- The attention mechanism calculates **relevance scores** between the **query** of the current token and the **keys** of all other tokens.
- The dot product between `Query` and `Key` matrices determines these scores.
- A **softmax operation** normalizes the scores into probabilities.

### **Weighted Sum of Values**
- The computed attention scores are used to weight the **Value** matrix.
- The output is an enriched representation of the token, integrating contextual information from relevant tokens in the sequence.

---

## **2. Multi-Head Self-Attention**
- Instead of a single attention mechanism, multiple attention heads operate in parallel.
- Each head captures different relationships in the data.
- The outputs of all heads are combined into a single representation.
- This enables the model to consider multiple perspectives at once.

---

## **3. Grouped Attention Mechanism**
- Introduces `n_groups` and `n_attention_heads`, where attention heads are grouped to improve efficiency.
- Each group processes a subset of keys and values, reducing computational cost.

---

## **4. Sparse Attention for Efficiency**
- Standard Transformers apply **global autoregressive self-attention**, meaning each token attends to all previous tokens.
- **Sparse Attention** reduces complexity by restricting attention to a limited number of past tokens.
  - **Strided Sparse Attention:** Looks at every nth token.
  - **Fixed Sparse Attention:** Attends to a fixed number of past tokens.

---

## **5. Token-Level Masking and Attention**
- A token can only pay attention to previous tokens, ensuring autoregressive behavior.
- Illustrated by an upper triangular matrix, where a token at position `t` can only attend to tokens `{1, 2, ..., t}`.

---

## **6. Ring Attention for Scaling Context Length**
- Traditional attention mechanisms are limited by **GPU memory constraints**.
- **Ring Attention** distributes queries, keys, and values across multiple GPUs to extend the effective context length.
- This approach enables near **infinite context window** processing.

---

## **7. Transformer Model Architecture Insights**
- Model configurations include:
  - **Layers (Depth)**
  - **Hidden Dimension**
  - **Feed-Forward Network (FFN) Dimension**
  - **Attention Heads**
  - **Key/Value Heads**
  - **Vocabulary Size**
  - **Activation Function (e.g., SwiGLU)**
  - **Position Encoding (e.g., RoPE - Rotary Position Embeddings)**

---

This summary covers **key attention optimizations**, **multi-head attention**, **sparse computation techniques**, and **scalability solutions** that improve Transformer efficiency. It provides an **intuitive understanding of attention mechanisms** while also linking to **GPU memory optimizations and large-scale context handling**.


# DAY 4
# DAY 5
# DAY 6
# DAY 7



## **📌 Projects**
- 📄 **LLM-Powered PDF Chatbot (RAG + AI Agent)**  
- ✉️ **Automated Email Response Bot**  
- 📈 **Financial News Analysis & Sentiment Extraction**  
- 💻 **AI Agent for Code Review & Debugging**  

## **🎯 Goals**
✅ Fully prepare for the OpenAI interview  
✅ Develop real-world AI projects  
✅ Gain expertise in LLM, RAG, and AI Agents  
✅ Build a strong portfolio on GitHub  

🚀 **Are you ready to join this journey?** 😎  
📌 **Follow along and star the repo!**
