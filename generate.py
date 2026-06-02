# =========================================
# generate_peptides_teacher_full_with_scoring.py
# Diffusion-based Teacher model for peptide generation
# Uses ESM2 embeddings, diffusion sampling, and hybrid scoring
# Generates multiple peptides per protein and ranks them
# =========================================

import os, math, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from collections import Counter
import esm

# ---------------- Device ----------------
# Select GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧩 Device:", device)

# ---------------- Settings ----------------
# Amino acid vocabulary (20 standard amino acids)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
PAD_IDX = len(AA_LIST)
vocab_size = len(AA_LIST) + 1

# Dimensionality settings
token_dim = 320        # peptide latent/token embedding dimension
prot_dim = 320         # protein embedding dimension

# Diffusion parameters
T = 30                 # number of diffusion timesteps

# Sampling parameters
TEMPERATURE = 1.0
TOP_K = 10
TOP_P = 0.95
N_SAMPLES = 5          # number of generated peptides per protein

# Hybrid scoring weights
# contrastive similarity + fluency (PPL) + AA distribution balance
ALPHA, BETA, GAMMA = 0.7, 0.3, 0.5

# ---------------- Paths ----------------
# Model checkpoints and data paths
TEACHER_MODEL_PATH = "/content/drive/MyDrive/pep_project/best_teacher_diffusion.pt"
PRETRAIN_PATH      = "/content/drive/MyDrive/pep_project/best_decoder_pretrain.pt"
TEST_CSV_PATH      = "/content/drive/MyDrive/pep_project/test.csv"
TEST_EMB_DIR       = "/content/drive/MyDrive/pep_project/emb-final/test"
OUTPUT_CSV         = "/content/drive/MyDrive/pep_project/1-generated_peptides_teacher_scored.csv"

# ---------------- Helper Functions ----------------
# Convert token indices back to amino acid sequences
def tokens_to_seqs(token_matrix):
    if torch.is_tensor(token_matrix):
        token_matrix = token_matrix.cpu().numpy()
    if token_matrix.ndim == 1:
        token_matrix = token_matrix[None,:]
    seqs = []
    for row in token_matrix:
        s = [AA_LIST[int(idx)] for idx in row if int(idx)<len(AA_LIST)]
        seqs.append("".join(s))
    return seqs[0] if len(seqs)==1 else seqs

# Top-p (nucleus) sampling for categorical distributions
def top_p_sampling(probs, top_p=0.95):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > top_p
    sorted_probs[mask] = 0
    if sorted_probs.sum() == 0:
        sorted_probs = torch.ones_like(sorted_probs)
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_idx[idx].item()

# Cosine noise schedule for diffusion process
def cosine_beta_schedule(T, s=0.008):
    steps = T+1
    t = torch.linspace(0,T,steps)/T
    alphas_cumprod = torch.cos((t+s)/(1+s)*math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas,1e-4,0.9999)

# Score measuring how close AA distribution of a sequence is to target distribution
def aa_distribution_score(seq, target_distribution):
    counter = Counter(seq)
    total = len(seq) if len(seq)>0 else 1
    score = 0.0
    for aa in AA_LIST:
        p_seq = counter.get(aa,0)/total
        p_target = target_distribution.get(aa,0)
        score += abs(p_seq - p_target)
    return 1.0 - score

# ---------------- Models ----------------
# Sinusoidal + learnable timestep embedding for diffusion
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, T):
        super().__init__()
        inv_freq = 1./(10000**(torch.arange(0,dim,2).float()/dim))
        self.register_buffer('inv_freq',inv_freq)
        self.learnable = nn.Embedding(T, dim)
    def forward(self, t):
        t = t.float()
        sinusoid = torch.cat([torch.sin(t[:,None]*self.inv_freq[None,:]),
                              torch.cos(t[:,None]*self.inv_freq[None,:])], dim=-1)
        return sinusoid + self.learnable(t.long())

# Transformer-based denoiser conditioned on protein embedding
class TransformerDenoiser(nn.Module):
    def __init__(self, token_dim, prot_dim, nhead=4, nlayer=3):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(token_dim, T)
        enc = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayer)
        self.out = nn.Linear(token_dim, token_dim)
        self.gamma_proj = nn.Linear(prot_dim, token_dim)
        self.beta_proj  = nn.Linear(prot_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.log_var_pep = nn.Linear(prot_dim, token_dim)

    # Predict noise given current noisy peptide, timestep, and protein embedding
    def forward(self, xt, t, prot_emb):
        B,L,D = xt.shape
        prot_mean = prot_emb.mean(1)
        gamma = self.gamma_proj(prot_mean).unsqueeze(1)
        beta  = self.beta_proj(prot_mean).unsqueeze(1)
        xt_mod = self.layer_norm(gamma*xt + beta)
        noise_pep = torch.randn_like(xt_mod)*torch.exp(0.5*self.log_var_pep(prot_mean)).unsqueeze(1)
        t_emb = self.time_emb(t).unsqueeze(1).expand(B,L,D)
        return self.out(self.encoder(xt_mod + t_emb + noise_pep))

# Transformer decoder mapping latent tokens to amino acid logits
class TransformerDecoder(nn.Module):
    def __init__(self, token_dim, vocab_size, nhead=8, nlayer=4, max_len=512):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, token_dim))
        enc = nn.TransformerEncoderLayer(d_model=token_dim,nhead=nhead,batch_first=True)
        self.transformer = nn.TransformerEncoder(enc,nlayer)
        self.out = nn.Linear(token_dim,vocab_size)
    def forward(self,x):
        B,L,D = x.shape
        x = x + self.pos_emb[:, :L, :]
        x = self.transformer(x)
        return self.out(x)

# Full diffusion teacher model
class DiffusionTeacher(nn.Module):
    def __init__(self, vocab_size, token_dim, prot_dim, T):
        super().__init__()
        self.decoder = TransformerDecoder(token_dim, vocab_size)
        self.denoiser = TransformerDenoiser(token_dim, prot_dim)
        betas = cosine_beta_schedule(T)
        alphas = 1.-betas
        alphas_bar = torch.cumprod(alphas,dim=0)
        self.register_buffer('betas',betas.to(device))
        self.register_buffer('alphas',alphas.to(device))
        self.register_buffer('alphas_bar',alphas_bar.to(device))
        self.T = T

    def predict_noise(self,xt,t,prot_emb):
        return self.denoiser(xt,t,prot_emb)

    # Reverse diffusion sampling to generate peptide tokens
    @torch.no_grad()
    def sample(self, prot_emb, L, temperature=1.0, topk=10, topp=0.95):
        x = torch.randn(1,L,token_dim,device=device)
        for t in reversed(range(1,self.T)):
            t_tensor = torch.tensor([t],device=device)
            eps = self.predict_noise(x,t_tensor,prot_emb)
            a = self.alphas[t]
            ab = self.alphas_bar[t]
            x = (1/torch.sqrt(a))*(x - ((1-a)/torch.sqrt(1-ab))*eps)
        logits = self.decoder(x)/temperature

        # Token sampling step
        tokens = []
        for i in range(L):
            token_probs = F.softmax(logits[0,i],dim=-1)
            topk_probs, topk_idx = torch.topk(token_probs, min(topk, token_probs.size(0)))
            topk_probs = topk_probs / (topk_probs.sum()+1e-12)
            idx = top_p_sampling(topk_probs, topp)
            tokens.append(topk_idx[idx].item())
        return torch.tensor(tokens, device=device), logits

# ---------------- Load Models ----------------
# Load trained diffusion teacher
teacher = DiffusionTeacher(vocab_size, token_dim, prot_dim, T).to(device)
teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH,map_location=device))
teacher.eval()
print("✅ Teacher loaded")

# Load ESM2 model for contrastive scoring
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

# Projection layer to match ESM embedding dimension
esm_proj = nn.Linear(esm_model.embed_dim, token_dim).to(device)
if os.path.exists(PRETRAIN_PATH):
    pretrain = torch.load(PRETRAIN_PATH,map_location=device)
    esm_proj.load_state_dict(pretrain["esm_proj"])
print("✅ ESM loaded")

# Embed a peptide sequence using ESM2
@torch.no_grad()
def esm_embed_seq(seq):
    batch = [("seq",seq)]
    _,_,tokens = batch_converter(batch)
    tokens = tokens.to(device)
    out = esm_model(tokens,repr_layers=[6])
    rep = out["representations"][6].mean(1)
    return esm_proj(rep)

# ---------------- Load Test Data ----------------
# Precomputed protein and peptide embeddings
prot_embs = torch.load(os.path.join(TEST_EMB_DIR,"test_prot_emb_000.pt"),map_location='cpu')
pep_embs  = torch.load(os.path.join(TEST_EMB_DIR,"test_pep_emb_000.pt"),map_location='cpu')

# Load test CSV
df = pd.read_csv(TEST_CSV_PATH)
results = []

# Compute global amino acid distribution from target peptides
all_targets = df['Binder'].tolist()
concat_targets = "".join([str(x) for x in all_targets if isinstance(x,str)])
aa_counter = Counter(concat_targets)
total_aa = sum(aa_counter.values()) if aa_counter else 1
aa_dist = {aa: aa_counter.get(aa,0)/total_aa for aa in AA_LIST}

# ---------------- Generation Loop ----------------
# For each protein:
# 1. Generate multiple peptide candidates
# 2. Score each candidate
# 3. Rank and store results
for i,row in tqdm(df.iterrows(), total=len(df)):
    prot_emb = prot_embs[i].unsqueeze(0).to(device)
    target_seq = str(row.get("Binder","") or "")
    L = max(len(target_seq),1)
    target_emb = pep_embs[i].mean(0).to(device)

    all_samples = []

    for s in range(N_SAMPLES):
        tokens, logits = teacher.sample(prot_emb,L,temperature=TEMPERATURE, topk=TOP_K, topp=TOP_P)
        seq = tokens_to_seqs(tokens)

        # Contrastive similarity score (ESM-based)
        try:
            gen_emb = esm_embed_seq(seq)
            contrastive_score = F.cosine_similarity(
                F.normalize(gen_emb,dim=-1),
                F.normalize(target_emb,dim=-1),
                dim=-1
            ).item()
        except Exception:
            contrastive_score = 0.0

        # Perplexity of generated sequence under decoder
        target_ids = [AA_LIST.index(a) if a in AA_LIST else PAD_IDX for a in seq]
        if len(target_ids) < logits.shape[1]:
            target_ids += [PAD_IDX]*(logits.shape[1]-len(target_ids))
        target_ids = torch.tensor(target_ids, device=device)
        ce_final = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=PAD_IDX,
            reduction='mean'
        )
        ppl_final = math.exp(ce_final.item())

        # Baseline unigram perplexity normalization
        eps_prob = 1e-9
        log_probs = [math.log(max(aa_dist.get(a, eps_prob), eps_prob)) for a in seq]
        baseline_ce = -sum(log_probs)/len(log_probs) if log_probs else -math.log(eps_prob)
        baseline_ppl = math.exp(baseline_ce)
        ppl_normalized = ppl_final / (baseline_ppl+1e-12)

        # Amino acid distribution balance score
        aa_score = aa_distribution_score(seq, aa_dist)

        # Final hybrid score
        final_score = ALPHA*contrastive_score + BETA*(1 - min(ppl_normalized/10,1.0)) + GAMMA*aa_score

        all_samples.append({
            "seq": seq,
            "contrastive": contrastive_score,
            "ppl_final": ppl_final,
            "ppl_normalized": ppl_normalized,
            "aa_score": aa_score,
            "final_score": final_score
        })

    # Rank generated peptides by final score
    all_samples.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, sample in enumerate(all_samples, start=1):
        results.append({
            "Index": i,
            "Receptor Sequence": row.get("Receptor Sequence",""),
            "Target Peptide": target_seq,
            "Generated Peptide": sample["seq"],
            "Rank": rank,
            "Contrastive Score": sample["contrastive"],
            "PPL Final": sample["ppl_final"],
            "PPL Normalized": sample["ppl_normalized"],
            "AA Balance": sample["aa_score"],
            "Final Score": sample["final_score"]
        })

    best = all_samples[0]
    print(f"[{i+1}/{len(df)}] Best Generated: {best['seq']} | Final Score={best['final_score']:.4f} | PPL_norm={best['ppl_normalized']:.3f}")

# ---------------- Save CSV ----------------
# Save all generated peptides and scores
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT_CSV,index=False)
print("✅ Generated peptides saved →", OUTPUT_CSV)
