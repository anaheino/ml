import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from BasicTokenizer import BasicTokenizer

batch_size = 32 #64 # how many parallel processing batches
block_size = 64 #256 # max size of context
max_iters = 5000 # 3000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters  = 200
n_embd =  192 # # 384
n_head =  3 # 6   #  384 / 6 = 64.0 
n_layer = 3 # 6
dropout = 0.10 # 0.2
default_gravity_strength = 5.25
default_gravity_on = True
default_gravity_target = torch.full((batch_size, block_size, block_size), block_size//2, dtype=torch.float32)
torch.manual_seed(1337)



with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('tokenizer.pkl', 'rb') as f:
    vocab, merges = pickle.load(f)

tokenizer = BasicTokenizer()
tokenizer.vocab = vocab
tokenizer.merges = merges
encode = lambda s: tokenizer.encode(s)  # encode using the trained BPE tokenizer
decode = lambda l: tokenizer.decode(l)  # decode using the trained BPE tokenizer
vocab_size = 276

data = torch.tensor(encode(text), dtype=torch.long) # convert to tensor for handling
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def toDevice(x, y):
    return x.to(device), y.to(device)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    # generating batch size of random offsets
    # and stacking
    x = torch.stack([data[i:i+block_size] for i in ix])
    # answers, lrn2gs
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return toDevice(x, y)


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


# This actually implements the whole
# key, value, query system that is at the heart of self attention
class Head(nn.Module):
    """ one head of self attention """
    def __init__(self, head_size):
        super().__init__()
        # näis layereis ei yleesä oo biasia, konventio
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # täs tehää se [1,0,0,0],[1,1,0,0]... matriisi
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        # Gravity well configuration
        # Example initial positions
        self.gravity_target = nn.Parameter(default_gravity_target)  
        self.gravity_strength = default_gravity_strength 
        self.gravity_on = default_gravity_on

    def compute_gravity_term(self, x, gravity_strength):
        """
        Compute gravity term for a tensor.
        
        Args:
            x (tensor): target tensor (batch_size, seq_len, feature_dim)..
            gravity_strength (float): Strength of the gravity pull.
        Returns:
            torch.Tensor: Gravity term tensor of shape `shape`.
        """
        B,T,C = x.shape
        # Adjust based on GPU capacity
        gravity_term = torch.zeros((1, T, 1), device=device)
        gravity_target = self.gravity_target.view(-1, 1, 1).expand(-1, T, 1)
        for start in range(0, T, block_size):
            end = min(start + block_size, T)
            positions_chunk = torch.arange(start, end, device=device).view(1, -1, 1)
            distances_chunk = (positions_chunk - gravity_target).abs()
            distances_sq_chunk = distances_chunk * distances_chunk
            gravity_term[:, start:end, :] = -gravity_strength / (distances_sq_chunk + 1e-6).sum(dim=0, keepdim=True)
        # 1. Convert gravity_target to a tensor
        # 2. Compute gravity contributions for all targets
        # 3. Sum contributions from all targets
        # positions = torch.arange(T, device=device).view(1, T, 1)  # Shape: (1, T, 1)
        # Assuming you want to apply gravity well at positions 64, 256, and 192 across the sequence
        # gravity_target = self.gravity_target.view(-1, 1, 1).expand(-1, T, 1)  
        # Shape: [1, T, 1]
        # distances = (positions - gravity_target).abs()  # Shape: (num_targets, T, 1)  
        # Gravity only affects sequence positions
        # gravity_contributions = -gravity_strength / (distances**2 + 1e-6)
        # gravity_term = gravity_contributions.sum(dim=0)  # Shape: (T, 1)
        return gravity_term
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B,T,C
        q = self.query(x) # B,T,C
        v = self.value(x)  # B,T,C
        # compute attention scores, aka affinities
        # C**-0.5 on käytännössä neliöjuuri 
        # (B, T, C) @ (B, C, T) -> (B, T, T), matrix multiplication
        weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # Optional: Add gravity well bias to attention scores
        if self.gravity_on:
            gravity_term = self.compute_gravity_term(x, self.gravity_strength)
            if torch.rand(1, device=device).item() < 0.5:
                weight += gravity_term
        # (T, T)
        # Incorporate gravity term into attention scores
        # decoder block
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        weight = F.softmax(weight, dim=-1)  # Normalize to get attention weights
        weight = self.dropout(weight)

        out = weight @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    

# Multithreading, basically.
class MultiThreadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # embeddings in this case vocalbulary,
        # each head filtered here together
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        # don't overfit 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Channels concatted, last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        # projection = linear transformation for result 
        # re-expansion back to fit the matrix size 
        return self.dropout(self.projection(out))

# Eli headeilla haetaan huomio
# sitten tällä annetaan tokeneille aikaa miettiä
# mitä huomio tarkoittaa, eli await multithreadeille
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(n_embd, n_embd * 4), 
            # ensin embeddetetaan,
            # muutetaan 4 kertaiseksi channel sizen
            # (dimensioiden) mukaan - korostetaan featureita
            nn.ReLU(),
            # ReLu = max([n]) ?? 0 
            # muutetaan negatiiviset arvot aiemmasta 
            # groupataan neuronit
            # lineaarisesta layerista max tai 0.
            # tää on sama kuin projection,
            # myös samalla muunnetaan takaisin alkuperäiseen muotoon
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class GatedFeedForward(nn.Module):
    """FeedForward with a gating mechanism."""
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.gate = nn.Linear(n_embd, n_embd)  # Gate mechanism
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Regular feed-forward path
        hidden = self.activation(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)

        # Gating mechanism
        gate = self.sigmoid(self.gate(x))  # Learn which parts to keep
        gated_output = gate * hidden + (1 - gate) * x  # Combine with input

        return gated_output


class Block(nn.Module):
    """Transformer block: communcation followed by computation"""

    def __init__(self, n_embd, n_head):
    
        # n_embd = embedding_dimension,
        # n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        # Self attention heads, eli lisätään 
        # n:lle (k,v,q) nodeille async parallel processing
        self.sa = MultiThreadAttention(n_head, head_size) 
        # käytännössä await, eli combine results
        # GatedFeedForward(n_embd)
        self.ffwd = FeedForward(n_embd)
        # normalizing values across rows, normalisointikerros
        self.ln1 = nn.LayerNorm(n_embd) 
        # normalisointi per tokeni,
        # joka normalisoi sen featuret
        self.ln2 = nn.LayerNorm(n_embd) 

    def forward(self, x):
        # skip connection architecture, eli "viereinen tie",
        # alussa on huono, mutta herää treenauksessa.
        # x -> return f(g(x)), f = collect, g = async think nx  
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 layeri josta saadaa vocab size, 
        # 2 layeri jost saadaa niide positio,
        # 3 niiden arvioitu lopputulos
        # embedding table => 
        # embed((getToken(sana)) -> vector([[1,0,0,1..],..])
        # HUOM! POSITIO + TOKEN EMBEDDING ==> relevanssi,
        # kuinka usein sana n seuraa sanaa j
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # we embed the values itself, vocab_size x 32
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # we embed the position of the values block_size x 32
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C), eli x sisältää ei pelkästään tokenit, mutta myös niiden sijainnit, missä ne esiintyy
        x = self.blocks(x) # (B,T,C) apply head of self-attention
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            # Stretching the tensors to work with F.cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)
            # lisää focal loss tai joku toinen, 
            # arvioi myös sillä, exponenttisumma juures?
            # kännis ja läbäl?
            loss = F.cross_entropy(logits, targets) # Just to use torch, it requires input in this form

        return logits, loss 
    
    # logits = unnormalized score for next character in the sequence 
    # loss = accuracy
    # Idx = (B, T) array of indices in the current context
    # current context, job of generate = B, T + 1, B, T + 2, B,T + 3 
    # so it continues generation of B and C in the Time dimension 
    # think of it as expanding a matrix mathematically n+rows n+cols
    # even though that's not exactly what's happening
    # When time is n+1, what is b and c?
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, elem <= block_size
            batch_to_process = idx[:, -block_size:] 
            logits, loss = self(batch_to_process)
            # for now only using 1, so n -> n+1, and n+1 -> n+2
            # instead of n, n+1 -> n+2 -> (B, C)
            # pluck out the last element in time dim/channel
            # because that's what comes next
            logits = logits[:, -1, :] 
            # softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # e.g. probability of each side of n-size dice being rolled, 
            # filtered through the -inf vector array
            idx_next = torch.multinomial(probs, num_samples=1) 
            # (B, 1), in each of the batch dimensions 
            # we have 1 prediction what comes next
            # append sampled index to running sequence
            # (B, T+1) dim=1 is the time dimension
            # C is the feature vector, channels
            # example feature: (111,215,34) for RGB values.
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx


model = GPT()
m = model.to(device)
# does the job that SGD does, finds local minima
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
torch.save(model.state_dict(), "model_gravity_well.pth")