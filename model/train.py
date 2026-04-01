import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16
block_size = 32          # now: max number of *tokens* (words) in context
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
conversation_memory = []
MAX_MEMORY = 5

# ------------

torch.manual_seed(1337)

# load text
with open('data/data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# --- Build Q/A pairs for retrieval ---
qa_pairs = []
lines = [l.strip() for l in text.splitlines() if l.strip()]

current_q = None
for line in lines:
    if line.startswith("Q:"):
        current_q = line[2:].strip()
    elif line.startswith("A:") and current_q is not None:
        answer = line[2:].strip()
        qa_pairs.append((current_q, answer))
        current_q = None

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# build embeddings for all questions
qa_embeddings = []
for q, a in qa_pairs:
    emb = embedder.encode(q, convert_to_tensor=True)
    qa_embeddings.append((emb, a))

# -------------------------------------

# simple word-level tokenizer
# split on whitespace; keep punctuation attached (good enough for now)
words = text.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

stoi = { w:i for i,w in enumerate(vocab) }
itos = { i:w for i,w in enumerate(vocab) }

def encode(s: str):
    return [stoi[w] for w in s.split() if w in stoi]

def decode(ids):
    return ' '.join([itos[i] for i in ids])

# turn entire corpus into token ids
data = torch.tensor(encode(text), dtype=torch.long)

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def question_similarity(q1: str, q2: str) -> float:
    w1 = set(q1.lower().replace('?', '').split())
    w2 = set(q2.lower().replace('?', '').split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def retrieve_best_answer(user_question, threshold=0.4):
    user_emb = embedder.encode(user_question, convert_to_tensor=True)

    best_score = -1
    best_answer = None

    for emb, answer in qa_embeddings:
        score = F.cosine_similarity(user_emb, emb, dim=0).item()
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= threshold:
        return best_answer
    return None


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model: start with a single dummy token
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

print("\nChat mode activated. Type 'quit' to exit.\n")

while True:
    user = input("You: ")
    if user.lower() == "quit":
        break

    # store user message
    conversation_memory.append(("user", user))
    if len(conversation_memory) > MAX_MEMORY:
        conversation_memory.pop(0)

    # 1) Try retrieval
    retrieved = retrieve_best_answer(user)
    if retrieved is not None:
        print("Cabbage person:", retrieved)
        conversation_memory.append(("bot", retrieved))
        continue

    # 2) If retrieval fails, use generative model
    tokens = [w for w in user.split() if w in stoi]
    if len(tokens) == 0:
        print("Bot: (I don't know these words yet.)")
        continue

    context = torch.tensor([stoi[w] for w in tokens], dtype=torch.long, device=device)[None, :]
    out = model.generate(context, max_new_tokens=50)[0].tolist()
    response = decode(out)

    print("Bot:", response)
    conversation_memory.append(("bot", response))

