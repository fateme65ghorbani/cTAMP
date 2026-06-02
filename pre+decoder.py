# ============================================
# Pretraining Transformer Decoder on Precomputed ESM Embeddings
# Full pipeline:
# - Load precomputed protein embeddings and peptide tokens
# - TransformerDecoder learns to map protein embeddings → peptide tokens
# - Validation with early stopping and LR scheduler
# ============================================
import torch, gc, time, glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- Device ----------------
# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧩 Device:", device)

# ---------------- Settings ----------------
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"  # Amino acids
PAD_IDX = len(AA_LIST)
vocab_size = len(AA_LIST) + 1

# Embedding dimensions
token_dim = 320      # ESM output dim
prot_dim  = 320

# Training hyperparameters
batch_size = 8
epochs = 50
lr = 1e-4
patience = 12

# Save path for best model
best_model_path = "/content/drive/MyDrive/pep_project/best_decoder_pretrain.pt"

# ---------------- ESM Projection ----------------
# No ESM model needed; use precomputed embeddings
esm_proj = nn.Linear(prot_dim, token_dim).to(device)

# ---------------- Dataset ----------------
# Dataset loads precomputed protein embeddings and peptide token sequences
class BatchEmbeddingDataset(Dataset):
    def __init__(self, prot_files, pep_token_files):
        self.prot_batches = [torch.load(f, map_location='cpu') for f in prot_files]
        self.pep_token_batches = [torch.load(f, map_location='cpu') for f in pep_token_files]
        self.lengths = [len(t) for t in self.pep_token_batches]
        self.cum_lengths = torch.cumsum(torch.tensor(self.lengths), dim=0)

    def __len__(self):
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx):
        file_idx = int((self.cum_lengths > idx).nonzero()[0])
        inner_idx = idx if file_idx == 0 else idx - int(self.cum_lengths[file_idx-1])
        prot_emb = self.prot_batches[file_idx][inner_idx].float()     # Protein embedding (L, 320)
        pep_tokens = self.pep_token_batches[file_idx][inner_idx].long() # Peptide tokens (T)
        return prot_emb, pep_tokens

# Collate function with padding for batch processing
def collate_fn(batch):
    prots, pep_tokens = zip(*batch)
    B = len(prots)

    # -------- Pad protein embeddings --------
    Lp = max(p.shape[0] for p in prots)
    Dp = prots[0].shape[1]
    prot_pad = torch.zeros(B, Lp, Dp, device=device)
    for i, p in enumerate(prots):
        prot_pad[i, :p.shape[0]] = p.to(device)

    # -------- Pad peptide token sequence --------
    Ltok = max(len(t) for t in pep_tokens)
    pep_token_pad = torch.full((B, Ltok), PAD_IDX, dtype=torch.long, device=device)
    for i, t in enumerate(pep_tokens):
        pep_token_pad[i, :len(t)] = t.to(device)

    return prot_pad, pep_token_pad

# ---------------- Load dataset ----------------
train_prot_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/train_prot_emb_*.pt"))
train_pep_token_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/train_pep_token_*.pt"))

val_prot_files   = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/test_prot_emb_*.pt"))
val_pep_token_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/test_pep_token_*.pt"))

train_dataset = BatchEmbeddingDataset(train_prot_files, train_pep_token_files)
val_dataset   = BatchEmbeddingDataset(val_prot_files, val_pep_token_files)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn)

# ---------------- Transformer Decoder ----------------
# Maps protein embeddings → peptide token probabilities
class TransformerDecoder(nn.Module):
    def __init__(self, token_dim, vocab_size, nhead=8, nlayer=4, max_len=512, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, token_dim))  # Positional embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.out = nn.Linear(token_dim, vocab_size)

    def forward(self, x):
        B, L, D = x.shape
        x = x + self.pos_emb[:, :L, :]
        x = self.transformer(x)
        logits = self.out(x)
        return logits

decoder = TransformerDecoder(token_dim, vocab_size).to(device)

# ---------------- Helper: embed protein (mean pooling only) ----------------
# Convert protein embeddings to initial input for decoder
def esm_embed_prot(prot_pad):
    rep = prot_pad.mean(dim=1)  # Mean pooling along sequence length
    return esm_proj(rep)

# ---------------- Optimizer & Scheduler ----------------
opt = torch.optim.AdamW(
    list(decoder.parameters()) + list(esm_proj.parameters()),
    lr=lr, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=3
)

# ---------------- Training Loop ----------------
# Standard supervised training with cross-entropy loss
best_val = float("inf")
epochs_no_improve = 0

for epoch in range(1, epochs+1):

    # ----- Training -----
    decoder.train()
    tr_loss = 0.0

    for prot_pad, pep_token_pad in tqdm(train_loader, desc=f"Train {epoch}"):
        B = prot_pad.size(0)
        x_emb = esm_embed_prot(prot_pad)                # Embed proteins [B, 320]
        x_emb = x_emb.unsqueeze(1).expand(B, pep_token_pad.size(1), token_dim)  # Repeat for sequence length [B, T, 320]

        logits = decoder(x_emb)  # Forward pass

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            pep_token_pad.reshape(-1),
            ignore_index=PAD_IDX
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        tr_loss += loss.item()

    # ----- Validation -----
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for prot_pad, pep_token_pad in val_loader:
            B = prot_pad.size(0)
            x_emb = esm_embed_prot(prot_pad)
            x_emb = x_emb.unsqueeze(1).expand(B, pep_token_pad.size(1), token_dim)
            logits = decoder(x_emb)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                pep_token_pad.reshape(-1),
                ignore_index=PAD_IDX
            )
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch}: Train={tr_loss/len(train_loader):.4f} | Val={val_loss:.4f}")
    scheduler.step(val_loss)

    # ----- Checkpointing & Early Stopping -----
    if val_loss < best_val:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save({
            "decoder": decoder.state_dict(),
            "esm_proj": esm_proj.state_dict(),
        }, best_model_path)
        print("💾 Best model saved.")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("⏹️ Early stopping.")
        break

    gc.collect()
    torch.cuda.empty_cache()

print("🏁 Pretraining finished.")
