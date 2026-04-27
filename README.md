Text Generation using Transformer Architecture — NLP / Deep Learning

A GPT-style Transformer language model built entirely from scratch using TensorFlow/Keras — no HuggingFace, no pre-built transformer libraries. Every component (MultiHeadAttention, positional embeddings, TransformerBlock) is implemented as a custom Keras layer and trained on the Harry Potter corpus for next-word text generation.

💡 What Makes This Different
Most NLP projects fine-tune a pre-trained model with 3 lines of HuggingFace code. This project builds the attention mechanism from the ground up — including Q/K/V projections, scaled dot-product attention, multi-head splitting, residual connections, and layer normalization — the same core components that power GPT-style models.

📋 Table of Contents

Architecture Overview
Project Structure
How It Works
Dataset
Installation
Model Configuration
Text Generation
What's Missing vs ChatGPT
Tech Stack


Architecture Overview
Input Tokens (seq_len = 50)
        │
        ▼
┌──────────────────────────────────┐
│   Token And Position Embedding   │
│                                  │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│        TransformerBlock          │
│                                  │
│  ┌────────────────────────────┐  │
│  │    MultiHeadAttention      │  │
│  │                            │  │
│  └────────────────────────────┘  │
│           + Residual             │
│       LayerNormalization         │
│                                  │
│  ┌────────────────────────────┐  │
│  │   Feed-Forward Network     │  │
│  │                            │  │
│  └────────────────────────────┘  │
│           + Residual             │
│       LayerNormalization         │
└──────────────────────────────────┘
        │
        ▼
  Last token output 
        │
        ▼
  Dense(vocab_size, softmax)
        │
        ▼
  Next-word probability distribution
HyperparameterValueEmbedding dimension128Number of attention heads4Feed-forward dimension512Sequence length50 tokensOptimizerAdamLossCategorical CrossentropyEpochs10Batch size32

Project Structure
transformer-text-generation/
│
├── Transformer.ipynb    ← Full implementation notebook
│
├── data/
│   └── hp_1.txt                  ← Harry Potter & the Philosopher's Stone
│                                    (download separately — see Dataset section)
├── requirements.txt
└── README.md

How It Works
1. Tokenization & Sequence Building
The raw text is lowercased and tokenized using Keras Tokenizer. Sliding windows of 50 tokens are created — the first 50 tokens are the input, the 51st is the prediction target.
2. TokenAndPositionEmbedding
Each token is mapped to a 128-dimensional embedding vector. Separately, each position index (0 to seq_len−1) is mapped to another 128-dimensional embedding. The two are added together — giving the model both what the word is and where it appears in the sequence.
3. MultiHeadAttention (built from scratch)
Q = query_dense(x)     # "What am I looking for?"
K = key_dense(x)       # "What do I contain?"
V = value_dense(x)     # "What do I return?"


4. TransformerBlock
Wraps MultiHeadAttention with:

Dropout (rate=0.1) for regularization
Residual connections — output = LayerNorm(x + attention_output) — preventing vanishing gradients
Feed-forward network — Dense(512 → 128) applied position-wise
Second residual + LayerNorm after the FFN

5. Text Generation
Takes a seed phrase, encodes it, runs it through the model, picks the highest-probability next word, appends it, and repeats for next_words steps.
pythonseed_text = "harry looked at"
generated = generate_text(seed_text, next_words=50, max_sequence_len=51)
print(generated)

Dataset
Harry Potter and the Philosopher's Stone (text format)

Not included in this repository due to copyright.
Download from: Kaggle — Harry Potter Books



Installation
Option A — Local
bash# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/transformer-text-generation.git
cd transformer-text-generation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
Option B — Google Colab (recommended, no GPU setup needed)

Upload Transformer.ipynb to Google Colab
Enable GPU: Runtime → Change runtime type → T4 GPU
Run all cells



Model Configuration
You can tune these parameters in the notebook:
pythonembed_dim  = 128   # Embedding size — increase for richer representations
num_heads  = 4     # Attention heads — must divide embed_dim evenly
ff_dim     = 512   # Feed-forward layer width
seq_length = 50    # Input sequence length
EPOCHS     = 10    # Training epochs
BATCH_SIZE = 32    # Batch size

Tip: Training on GPU (Colab T4) takes ~5–10 minutes. On CPU expect 30–60 minutes.


Text Generation
Example output after 10 epochs (seed: "harry looked at"):
harry looked at the door and saw a small man in a long
cloak who was peering into the shop with great interest
the door opened and the little man stepped inside...
The model captures writing style, sentence rhythm, and character-consistent vocabulary — though without causal masking or stacked layers, long-range coherence is limited.

What's Missing vs ChatGPT
This project is an educational implementation. Here's how it compares to production LLMs:
FeatureThis ModelGPT / ChatGPTCausal masking❌ Sees full sequence✅ Can only see past tokensStacked layers1 TransformerBlock12–96 Transformer layersTokenizationWord-level Keras tokenizerByte-Pair Encoding (BPE)Training data1 book (~500KB)Hundreds of GB of textDecoding strategyGreedy argmaxTop-k, nucleus sampling, beam searchParameters~2M117M – 175B+
Understanding these gaps — and being able to articulate them — is the real learning outcome of this project.

Tech Stack
LibraryPurposeTensorFlow / KerasCustom layer building, model trainingNumPyArray operations, sequence manipulationKeras TokenizerText tokenization and vocabulary mapping
