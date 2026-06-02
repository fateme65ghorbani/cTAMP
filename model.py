# =========================================
# Diffusion Teacher (T=30) + ESM-2 + CE_tok + Contrastive + TransformerDecoder
# Full pipeline:
# - Dataset loading from precomputed embeddings
# - Diffusion-based teacher training
# - Validation with early stopping
# - Example inference (sampling)
# =========================================
import torch, gc, glob, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import esm
import matplotlib.pyplot as plt

# ---------------- Device ----------------
# Select computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧩 Device:", device)

# ---------------- Settings ----------------
# Amino acid vocabulary and padding
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
PAD_IDX = len(AA_LIST)
vocab_size = len(AA_LIST) + 1

# Embedding dimensions
token_dim = 320
prot_dim = 320

# Diffusion and training hyperparameters
T = 30
batch_size = 8
accum_steps = 1
epochs = 50
lr = 1e-4
patience = 12

# Paths for pretrained components and saving teacher
pretrain_path = "/content/drive/MyDrive/pep_project/best_decoder_pretrain.pt"
teacher_path = "/content/drive/MyDrive/pep_project/best_teacher_diffusion.pt"

# Loss weights
λ_diff, λ_ce, λ_con = 1.0, 0.05, 0.1

# ---------------- Load ESM-2 ----------------
# Load pretrained ESM-2 model (frozen)
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.eval().to(device)
for param in esm_model.parameters():
    param.requires_grad = False

batch_converter = alphabet.get_batch_converter()

# Projection layer to align ESM embedding dimension with token_dim
esm_proj = nn.Linear(esm_model.embed_dim, token_dim).to(device)

# ---------------- Cosine beta schedule ----------------
# Noise schedule for diffusion process
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)

# ---------------- Dataset ----------------
# Dataset that loads precomputed protein embeddings,
# peptide embeddings, and peptide token sequences from disk
class BatchEmbeddingDataset(Dataset):
    def __init__(self, prot_files, pep_emb_files, pep_token_files):
        self.prot_batches = [torch.load(f, map_location='cpu') for f in prot_files]
        self.pep_emb_batches = [torch.load(f, map_location='cpu') for f in pep_emb_files]
        self.pep_token_batches = [torch.load(f, map_location='cpu') for f in pep_token_files]
        self.lengths = [len(t) for t in self.pep_token_batches]
        self.cum_lengths = torch.cumsum(torch.tensor(self.lengths), dim=0)
        print(f"📦 Loaded {len(self.pep_token_batches)} files → total samples: {int(self.cum_lengths[-1])}")

    def __len__(self):
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx):
        file_idx = int((self.cum_lengths > idx).nonzero()[0])
        inner_idx = idx if file_idx == 0 else idx - int(self.cum_lengths[file_idx-1])
        prot = self.prot_batches[file_idx][inner_idx].float()
        pep_emb = self.pep_emb_batches[file_idx][inner_idx].float()
        pep_tokens = self.pep_token_batches[file_idx][inner_idx].long()
        return prot, pep_emb, pep_tokens

# Custom collate function with padding for variable-length sequences
def collate_fn(batch):
    prots, pep_embs, pep_tokens = zip(*batch)
    B = len(prots)

    Lpep = max(e.shape[0] for e in pep_embs)
    De = pep_embs[0].shape[1]
    pep_emb_pad = torch.zeros(B, Lpep, De, device=device)
    for i,e in enumerate(pep_embs):
        pep_emb_pad[i,:e.shape[0],:] = e.to(device)

    Lp = max(p.shape[0] for p in prots)
    Dp = prots[0].shape[1]
    prot_pad = torch.zeros(B, Lp, Dp, device=device)
    for i,p in enumerate(prots):
        prot_pad[i,:p.shape[0],:] = p.to(device)

    Ltok = max(len(t) for t in pep_tokens)
    pep_token_pad = torch.full((B,Ltok), PAD_IDX, dtype=torch.long, device=device)
    for i,t in enumerate(pep_tokens):
        pep_token_pad[i,:len(t)] = t.to(device)

    return prot_pad, pep_emb_pad, pep_token_pad

# ---------------- Load dataset ----------------
# Training and validation datasets use precomputed embedding batches
train_prot_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/train_prot_emb_*.pt"))
train_pep_emb_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/train_pep_emb_*.pt"))
train_pep_token_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/train_pep_token_*.pt"))

val_prot_files   = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/test_prot_emb_*.pt"))
val_pep_emb_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/test_pep_emb_*.pt"))
val_pep_token_files = sorted(glob.glob("/content/drive/MyDrive/pep_project/emb-final/test_pep_token_*.pt"))

train_dataset = BatchEmbeddingDataset(train_prot_files, train_pep_emb_files, train_pep_token_files)
val_dataset   = BatchEmbeddingDataset(val_prot_files, val_pep_emb_files, val_pep_token_files)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---------------- Time Embedding ----------------
# Sinusoidal + learnable timestep embedding for diffusion
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, T):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.learnable = nn.Embedding(T, dim)
    def forward(self, t):
        t = t.float()
        sinusoid = torch.cat([torch.sin(t[:,None]*self.inv_freq[None,:]),
                              torch.cos(t[:,None]*self.inv_freq[None,:])], dim=-1)
        return sinusoid + self.learnable(t.long())

# ---------------- Transformer Denoiser ----------------
# Predicts diffusion noise conditioned on protein embedding
class TransformerDenoiser(nn.Module):
    def __init__(self, token_dim, prot_dim, nhead=4, nlayer=3, dropout=0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(token_dim, T)
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.out = nn.Linear(token_dim, token_dim)
        self.gamma_proj = nn.Linear(prot_dim, token_dim)
        self.beta_proj = nn.Linear(prot_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.log_var_pep = nn.Linear(prot_dim, token_dim)

    def forward(self, xt, t, prot_emb):
        B,L,D = xt.shape
        prot_mean = prot_emb.mean(1)
        gamma = self.gamma_proj(prot_mean).unsqueeze(1)
        beta  = self.beta_proj(prot_mean).unsqueeze(1)
        xt_mod = gamma*xt + beta
        xt_mod = self.layer_norm(xt_mod)
        noise_pep = torch.randn_like(xt_mod)*torch.exp(0.5*self.log_var_pep(prot_mean)).unsqueeze(1)
        xt_mod = xt_mod + noise_pep
        t_emb = self.time_emb(t).unsqueeze(1).expand(B,L,D)
        out = self.out(self.encoder(xt_mod+t_emb))
        return out

# ---------------- Transformer Decoder (Pretrained) ----------------
# Maps denoised latent representations to amino acid logits
class TransformerDecoder(nn.Module):
    def __init__(self, token_dim, vocab_size, nhead=8, nlayer=4, max_len=512, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.out = nn.Linear(token_dim, vocab_size)

    def forward(self, x):
        B,L,D = x.shape
        x = x + self.pos_emb[:, :L, :]
        x = self.transformer(x)
        logits = self.out(x)
        return logits

# ---------------- Diffusion Teacher ----------------
# Full teacher model combining diffusion denoiser and decoder
class DiffusionTeacher(nn.Module):
    def __init__(self, vocab_size, token_dim, prot_dim, T):
        super().__init__()
        self.decoder = TransformerDecoder(token_dim, vocab_size).to(device)
        self.denoiser = TransformerDenoiser(token_dim, prot_dim)
        betas = cosine_beta_schedule(T)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alphas_bar', alphas_bar.to(device))
        self.T = T

    # Forward diffusion (q)
    def q_sample(self, x0_emb, t):
        B,L,D = x0_emb.shape
        a_bar = self.alphas_bar[t].view(B,1,1)
        noise = torch.randn_like(x0_emb)
        xt = torch.sqrt(a_bar)*x0_emb + torch.sqrt(1.-a_bar)*noise
        return xt, noise

    # Noise prediction
    def predict_noise(self, xt, t, prot_emb):
        return self.denoiser(xt, t, prot_emb)

    # Reverse diffusion sampling
    @torch.no_grad()
    def sample(self, prot_emb, L, return_logits=False):
        device_ = prot_emb.device
        xt = torch.randn(1, L, token_dim, device=device_)
        for t in reversed(range(1, self.T)):
            eps = self.predict_noise(xt, torch.tensor([t], device=device_), prot_emb)
            a = self.alphas[t]
            ab = self.alphas_bar[t]
            xt = (1/torch.sqrt(a))*(xt - ((1-a)/torch.sqrt(1-ab))*eps)
        logits = self.decoder(xt)
        tokens = logits.argmax(-1)
        if return_logits:
            return tokens, logits
        return tokens

# ---------------- Load Pretrained TransformerDecoder ----------------
# Initialize teacher and load pretrained decoder and ESM projection
pretrain_ckpt = torch.load(pretrain_path, map_location=device)
teacher = DiffusionTeacher(vocab_size, token_dim, prot_dim, T).to(device)
teacher.decoder.load_state_dict(pretrain_ckpt["decoder"])
esm_proj.load_state_dict(pretrain_ckpt["esm_proj"])
print("✅ Pretrained TransformerDecoder loaded into teacher!")

# ---------------- Helper functions ----------------
# Convert token indices to amino acid sequences
def tokens_to_seqs(token_matrix):
    seqs = []
    token_matrix = token_matrix.cpu().numpy()
    for row in token_matrix:
        s = []
        for idx in row:
            if idx < len(AA_LIST): s.append(AA_LIST[int(idx)])
            else: break
        seqs.append("".join(s))
    return seqs

# Embed sequences using frozen ESM-2
def esm_embed_seqs(seq_list):
    labels = [str(i) for i in range(len(seq_list))]
    batch = list(zip(labels, seq_list))
    _,_,batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        out = esm_model(batch_tokens, repr_layers=[6])
        rep = out["representations"][6].mean(1)
    return esm_proj(rep)

# ---------------- Optimizer & Scheduler ----------------
# Optimize teacher and projection layer
opt = torch.optim.AdamW(
    list(teacher.parameters()) + list(esm_proj.parameters()),
    lr=lr,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

# ---------------- Training loop ----------------
# Includes diffusion loss, token CE loss, and contrastive loss
best_val = float('inf')
epochs_no_improve = 0
train_losses, val_losses = [], []

for epoch in range(1, epochs+1):
    teacher.train()
    tr_total=tr_diff=tr_ce=tr_con=0.0
    pbar = tqdm(train_loader, desc=f"🌀 Epoch {epoch} [Train]", leave=False)

    for step,(prot_pad,pep_emb_pad,pep_token_pad) in enumerate(pbar):
        B = prot_pad.size(0)
        Ltok = pep_token_pad.size(1)

        pooled = pep_emb_pad.mean(1)
        target_emb_proj = esm_proj(pooled.to(device)).detach()
        x0_single = target_emb_proj
        x0_emb = x0_single.unsqueeze(1).expand(B,Ltok,token_dim).to(device)
        prot_pad = prot_pad.to(device)
        pep_token_pad = pep_token_pad.to(device)

        t = torch.randint(1,teacher.T,(B,),device=device)
        xt, noise = teacher.q_sample(x0_emb,t)
        eps_pred = teacher.predict_noise(xt,t,prot_pad)
        loss_diff = F.mse_loss(eps_pred, noise)

        a_bar = teacher.alphas_bar[t].view(B,1,1)
        x0_pred = (xt - torch.sqrt(1.-a_bar)*eps_pred)/(torch.sqrt(a_bar)+1e-8)
        logits = teacher.decoder(x0_pred)

        loss_ce_token = F.cross_entropy(
            logits.view(-1,logits.size(-1)),
            pep_token_pad.view(-1),
            ignore_index=PAD_IDX
        )

        pred_tokens = torch.argmax(logits, dim=-1)
        pred_seqs = tokens_to_seqs(pred_tokens)

        with torch.no_grad():
            pred_emb = esm_embed_seqs(pred_seqs)

        pred_norm = F.normalize(pred_emb, dim=-1)
        target_norm = F.normalize(target_emb_proj, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        loss_con = (1.0 - cos_sim).mean()

        total = λ_diff*loss_diff + λ_ce*loss_ce_token + λ_con*loss_con
        (total/accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)

        if (step+1)%accum_steps==0:
            opt.step()
            opt.zero_grad()

        tr_total+=total.item()
        tr_diff+=loss_diff.item()
        tr_ce+=loss_ce_token.item()
        tr_con+=loss_con.item()

    # ---- Validation ----
    teacher.eval()
    val_total=val_diff=val_ce=val_con=0.0
    with torch.no_grad():
        for prot_pad,pep_emb_pad,pep_token_pad in val_loader:
            B = prot_pad.size(0)
            Ltok = pep_token_pad.size(1)

            pooled = pep_emb_pad.mean(1)
            target_emb_proj = esm_proj(pooled.to(device)).detach()
            x0_single = target_emb_proj
            x0_emb = x0_single.unsqueeze(1).expand(B,Ltok,token_dim).to(device)
            prot_pad = prot_pad.to(device)
            pep_token_pad = pep_token_pad.to(device)

            t = torch.randint(1,teacher.T,(B,),device=device)
            xt, noise = teacher.q_sample(x0_emb,t)
            eps_pred = teacher.predict_noise(xt,t,prot_pad)
            loss_diff = F.mse_loss(eps_pred, noise)

            a_bar = teacher.alphas_bar[t].view(B,1,1)
            x0_pred = (xt - torch.sqrt(1.-a_bar)*eps_pred)/(torch.sqrt(a_bar)+1e-8)
            logits = teacher.decoder(x0_pred)

            loss_ce_token = F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                pep_token_pad.view(-1),
                ignore_index=PAD_IDX
            )

            pred_tokens = torch.argmax(logits, dim=-1)
            pred_seqs = tokens_to_seqs(pred_tokens)
            pred_emb = esm_embed_seqs(pred_seqs)

            pred_norm = F.normalize(pred_emb, dim=-1)
            target_norm = F.normalize(target_emb_proj, dim=-1)
            cos_sim = (pred_norm * target_norm).sum(dim=-1)
            loss_con = (1.0 - cos_sim).mean()

            val_total += λ_diff*loss_diff.item() + λ_ce*loss_ce_token.item() + λ_con*loss_con.item()
            val_diff += loss_diff.item()
            val_ce += loss_ce_token.item()
            val_con += loss_con.item()

    scheduler.step(val_total)
    train_losses.append(tr_total/len(train_loader))
    val_losses.append(val_total/len(val_loader))

    print(f"\n📘 Epoch {epoch}/{epochs}")
    print(f"Train → Total={train_losses[-1]:.4f} | Diff={tr_diff/len(train_loader):.4f} | CE_tok={tr_ce/len(train_loader):.4f} | Con={tr_con/len(train_loader):.4f}")
    print(f"Valid → Total={val_losses[-1]:.4f} | Diff={val_diff/len(val_loader):.4f} | CE_tok={val_ce/len(val_loader):.4f} | Con={val_con/len(val_loader):.4f}")

    if val_total < best_val:
        best_val = val_total
        epochs_no_improve = 0
        torch.save(teacher.state_dict(), teacher_path)
        print("✅ Best model saved!")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= patience:
        print("⏹️ Early stopping triggered.")
        break

    gc.collect()
    torch.cuda.empty_cache()

# ---------------- Plot training ----------------
# Visualize training and validation loss curves
plt.figure(figsize=(10,5))
plt.plot(train_losses,label='train')
plt.plot(val_losses,label='val')
plt.legend(); plt.grid(True)
plt.show()
print("🏁 Training done.")

# ---------------- Example inference ----------------
# prot_seq = ["YOUR_PROTEIN_SEQUENCE"]
# prot_emb = esm_embed_seqs(prot_seq)
# tokens = teacher.sample(prot_emb, L=desired_peptide_length)
# seq = tokens_to_seqs(tokens)
# print(seq)
