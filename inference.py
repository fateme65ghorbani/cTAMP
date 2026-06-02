# ============================================
# single_protein_peptide_generation_verbose_first_only.py
# Generate multiple peptides for a single protein using Diffusion Teacher
# Uses ESM2 embeddings to score peptides
# Verbose debug prints only for first peptide
# ============================================

import os, math, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
import esm
import pandas as pd

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧩 [Debug] Device:", device)

# ---------------- Settings ----------------
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
PAD_IDX = len(AA_LIST)
vocab_size = len(AA_LIST)+1

token_dim = 320
prot_dim = 320
T = 30
TEMPERATURE = 1.0
TOP_K = 10
TOP_P = 0.95
N_SAMPLES = 30
ALPHA, BETA, GAMMA = 0.7, 0.3, 0.5
print(f"⚙️ [Debug] Settings: N_SAMPLES={N_SAMPLES}, Temperature={TEMPERATURE}, TopK={TOP_K}, TopP={TOP_P}")

# ---------------- Paths ----------------
TEACHER_MODEL_PATH = "/content/drive/MyDrive/pep_project/best_teacher_diffusion.pt"
PRETRAIN_PATH      = "/content/drive/MyDrive/pep_project/best_decoder_pretrain.pt"

# ---------------- Helper Functions ----------------
def tokens_to_seqs(token_matrix, verbose=False):
    if torch.is_tensor(token_matrix):
        token_matrix = token_matrix.cpu().numpy()
    if token_matrix.ndim == 1:
        token_matrix = token_matrix[None,:]
    seqs = []
    for row in token_matrix:
        s = [AA_LIST[int(idx)] for idx in row if int(idx)<len(AA_LIST)]
        seqs.append("".join(s))
    if verbose:
        print(f"🔹 [Debug] tokens_to_seqs: {seqs}")
    return seqs[0] if len(seqs)==1 else seqs

def top_p_sampling(probs, top_p=0.95, verbose=False):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > top_p
    sorted_probs[mask] = 0
    if sorted_probs.sum() == 0:
        sorted_probs = torch.ones_like(sorted_probs)
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1).item()
    if verbose:
        print(f"🔹 [Debug] top_p_sampling selected idx={sorted_idx[idx].item()}")
    return sorted_idx[idx].item()

def cosine_beta_schedule(T, s=0.008):
    steps = T+1
    t = torch.linspace(0,T,steps)/T
    alphas_cumprod = torch.cos((t+s)/(1+s)*math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas,1e-4,0.9999)

def aa_distribution_score(seq, target_distribution, verbose=False):
    counter = Counter(seq)
    total = len(seq) if len(seq)>0 else 1
    score = 0.0
    for aa in AA_LIST:
        p_seq = counter.get(aa,0)/total
        p_target = target_distribution.get(aa,0)
        score += abs(p_seq - p_target)
    if verbose:
        print(f"🔹 [Debug] aa_distribution_score: {score}")
    return 1.0 - score

# ---------------- Models ----------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, T):
        super().__init__()
        inv_freq = 1./(10000**(torch.arange(0,dim,2).float()/dim))
        self.register_buffer('inv_freq',inv_freq)
        self.learnable = nn.Embedding(T, dim)
        print("⏱️ [Debug] SinusoidalTimeEmbedding initialized")
    def forward(self, t):
        t = t.float()
        sinusoid = torch.cat([torch.sin(t[:,None]*self.inv_freq[None,:]),
                              torch.cos(t[:,None]*self.inv_freq[None,:])], dim=-1)
        return sinusoid + self.learnable(t.long())

class TransformerDenoiser(nn.Module):
    def __init__(self, token_dim, prot_dim, nhead=4, nlayer=3, dropout=0.1):
        super().__init__()
        print("🌀 [Debug] TransformerDenoiser initialized")
        self.time_emb = SinusoidalTimeEmbedding(token_dim, T)
        enc = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayer)
        self.out = nn.Linear(token_dim, token_dim)
        self.gamma_proj = nn.Linear(prot_dim, token_dim)
        self.beta_proj  = nn.Linear(prot_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.log_var_pep = nn.Linear(prot_dim, token_dim)
    def forward(self, xt, t, prot_emb, verbose=False):
        B,L,D = xt.shape
        prot_mean = prot_emb.mean(1) if prot_emb.dim()==3 else prot_emb
        gamma = self.gamma_proj(prot_mean).unsqueeze(1)
        beta  = self.beta_proj(prot_mean).unsqueeze(1)
        xt_mod = self.layer_norm(gamma*xt + beta)
        noise_pep = torch.randn_like(xt_mod)*torch.exp(0.5*self.log_var_pep(prot_mean)).unsqueeze(1)
        t_emb = self.time_emb(t).unsqueeze(1).expand(B,L,D)
        out = self.out(self.encoder(xt_mod + t_emb + noise_pep))
        if verbose:
            print(f"🔹 [Debug] TransformerDenoiser forward: out.shape={out.shape}")
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, token_dim, vocab_size, nhead=8, nlayer=4, max_len=512):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, token_dim))
        enc = nn.TransformerEncoderLayer(d_model=token_dim,nhead=nhead,batch_first=True)
        self.transformer = nn.TransformerEncoder(enc,nlayer)
        self.out = nn.Linear(token_dim,vocab_size)
        print("📝 [Debug] TransformerDecoder initialized")
    def forward(self,x):
        B,L,D = x.shape
        x = x + self.pos_emb[:, :L, :]
        x = self.transformer(x)
        return self.out(x)

class DiffusionTeacher(nn.Module):
    def __init__(self, vocab_size, token_dim, prot_dim, T):
        super().__init__()
        print("🧑‍🏫 [Debug] DiffusionTeacher initialized")
        self.decoder = TransformerDecoder(token_dim, vocab_size)
        self.denoiser = TransformerDenoiser(token_dim, prot_dim)
        betas = cosine_beta_schedule(T)
        alphas = 1.-betas
        alphas_bar = torch.cumprod(alphas,dim=0)
        self.register_buffer('betas',betas.to(device))
        self.register_buffer('alphas',alphas.to(device))
        self.register_buffer('alphas_bar',alphas_bar.to(device))
        self.T = T
    def predict_noise(self,xt,t,prot_emb, verbose=False):
        if verbose:
            print(f"🔹 [Debug] DiffusionTeacher predict_noise called")
        return self.denoiser(xt,t,prot_emb, verbose=verbose)
    @torch.no_grad()
    def sample(self, prot_emb, L, temperature=1.0, topk=10, topp=0.95, verbose=False):
        if verbose:
            print(f"🔹 [Debug] Sampling started: L={L}")
        x = torch.randn(1,L,token_dim,device=device)
        for t in reversed(range(1,self.T)):
            t_tensor = torch.tensor([t],device=device)
            eps = self.predict_noise(x,t_tensor,prot_emb, verbose=verbose)
            a = self.alphas[t]
            ab = self.alphas_bar[t]
            x = (1/torch.sqrt(a))*(x - ((1-a)/torch.sqrt(1-ab))*eps)
        logits = self.decoder(x)/temperature
        tokens = []
        for i in range(L):
            token_probs = F.softmax(logits[0,i],dim=-1)
            topk_probs, topk_idx = torch.topk(token_probs, min(topk, token_probs.size(0)))
            topk_probs = topk_probs / (topk_probs.sum()+1e-12)
            idx = top_p_sampling(topk_probs, topp, verbose=verbose)
            tokens.append(topk_idx[idx].item())
        if verbose:
            print(f"🔹 [Debug] Sampling finished: tokens={tokens}")
        return torch.tensor(tokens, device=device), logits

# ---------------- Load Teacher & Pretrained Decoder ----------------
teacher = DiffusionTeacher(vocab_size, token_dim, prot_dim, T).to(device)
teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH,map_location=device))
teacher.eval()
print("✅ [Debug] Teacher loaded")

esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()
esm_proj = nn.Linear(esm_model.embed_dim, token_dim).to(device)
if os.path.exists(PRETRAIN_PATH):
    pretrain = torch.load(PRETRAIN_PATH,map_location=device)
    esm_proj.load_state_dict(pretrain["esm_proj"])
print("✅ [Debug] ESM loaded")

@torch.no_grad()
def esm_embed_seq(seq, verbose=False):
    batch = [("seq",seq)]
    _,_,tokens = batch_converter(batch)
    tokens = tokens.to(device)
    out = esm_model(tokens,repr_layers=[6])
    rep = out["representations"][6].mean(1)
    if verbose:
        print(f"🔹 [Debug] esm_embed_seq computed embedding for seq={seq}")
    return esm_proj(rep)

@torch.no_grad()
def embed_protein(seq, verbose=False):
    batch = [("seq",seq)]
    _,_,tokens = batch_converter(batch)
    tokens = tokens.to(device)
    out = esm_model(tokens, repr_layers=[6])
    rep = out["representations"][6].mean(1)
    rep = esm_proj(rep)
    if verbose:
        print(f"🔹 [Debug] embed_protein computed embedding for protein length={len(seq)}")
    return rep

# ---------------- Single Protein Input ----------------
# Note: only the protein sequence is used as input for Diffusion conditions
protein_seq = "PSSSMADFRKFFAKAKHIVIISGAGVSAESGVPTFRGAGGYWRKWQAQDLATPLAFAHNPSRVWEFYHYRREVMGSKEPNAGHRAIAECETRLGKQGRRVVVITQNIDELHRKAGTKNLLEIHGSLFKTRCTSCGVVAENYKSPICPALSGKGAPEPGTQDASIPVEKLPRCEEAGCGGLLRPHVVWFGENLDPAILEEVDRELAHCDLCLVVGTSSVVYPAAMFAPQVAARGVPVAEFNTETTPATNRFRFHFQGPCGTTLPEALAXX"
target_pep_seq = "AVXCAX"  # Only used for final evaluation, not input to diffusion
L = len(target_pep_seq)
print(f"⚙️ [Debug] Protein length={len(protein_seq)}, target peptide length={L}")

aa_counter = Counter(target_pep_seq)
total_aa = sum(aa_counter.values()) if aa_counter else 1
aa_dist = {aa: aa_counter.get(aa,0)/total_aa for aa in AA_LIST}
print(f"🔹 [Debug] AA distribution for target peptide: {aa_dist}")

prot_emb = embed_protein(protein_seq, verbose=True)
target_emb = esm_embed_seq(target_pep_seq, verbose=True)

# ---------------- Generate Peptides ----------------
results = []
for s in range(N_SAMPLES):
    verbose = (s == 0)  # Only verbose for first sample
    if verbose:
        print(f"\n🧪 [Sample {s+1}/{N_SAMPLES}] Generating peptide...")
    tokens, logits = teacher.sample(prot_emb, L, temperature=TEMPERATURE, topk=TOP_K, topp=TOP_P, verbose=verbose)
    seq = tokens_to_seqs(tokens, verbose=verbose)

    # Only for evaluation: contrastive similarity with target peptide
    try:
        gen_emb = esm_embed_seq(seq, verbose=verbose)
        contrastive_score = F.cosine_similarity(F.normalize(gen_emb,dim=-1),
                                               F.normalize(target_emb,dim=-1),dim=-1).item()
    except Exception as e:
        if verbose:
            print(f"⚠️ [Debug] Contrastive score exception: {e}")
        contrastive_score = 0.0
    if verbose:
        print(f"🔹 [Debug] Contrastive score={contrastive_score:.4f}")

    # PPL score
    target_ids = [AA_LIST.index(a) if a in AA_LIST else PAD_IDX for a in seq]
    if len(target_ids) < logits.shape[1]:
        target_ids += [PAD_IDX]*(logits.shape[1]-len(target_ids))
    target_ids = torch.tensor(target_ids, device=device)
    ce_final = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=PAD_IDX, reduction='mean')
    ppl_final = math.exp(ce_final.item())
    if verbose:
        print(f"🔹 [Debug] PPL final={ppl_final:.4f}")

    # Baseline unigram PPL
    eps_prob = 1e-9
    log_probs = [math.log(max(aa_dist.get(a, eps_prob), eps_prob)) for a in seq]
    baseline_ce = -sum(log_probs)/len(log_probs) if log_probs else -math.log(eps_prob)
    baseline_ppl = math.exp(baseline_ce)
    ppl_normalized = ppl_final / (baseline_ppl+1e-12)
    if verbose:
        print(f"🔹 [Debug] PPL normalized={ppl_normalized:.4f}")

    # AA distribution score
    aa_score = aa_distribution_score(seq, aa_dist, verbose=verbose)

    # Hybrid final score
    final_score = ALPHA*contrastive_score + BETA*(1 - min(ppl_normalized/10,1.0)) + GAMMA*aa_score
    if verbose:
        print(f"🔹 [Debug] Final score={final_score:.4f}")

    results.append({
        "Generated Peptide": seq,
        "Contrastive Score": contrastive_score,
        "PPL Final": ppl_final,
        "PPL Normalized": ppl_normalized,
        "AA Balance": aa_score,
        "Final Score": final_score
    })

# ---------------- Sort and Save Results ----------------
results.sort(key=lambda x: x["Final Score"], reverse=True)
for rank, sample in enumerate(results[:10], start=1):
    print(f"🏆 Rank {rank}: {sample['Generated Peptide']} | Final Score: {sample['Final Score']:.4f}")

df_out = pd.DataFrame(results)
df_out.to_csv("/content/drive/MyDrive/pep_project/single_protein_generated_peptides_verbose_first_only.csv", index=False)
print("✅ [Debug] Saved all generated peptides to CSV")
