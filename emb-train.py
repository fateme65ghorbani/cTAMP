# =========================
# Imports
# Required libraries for tensor computation, file handling, data loading,
# progress visualization, and pretrained ESM models
# =========================
import torch, os, pandas as pd
from tqdm import tqdm
from esm import pretrained

# =========================
# Configuration & paths
# Define output directory and dataset paths
# =========================
save_dir = "/content/drive/MyDrive/pep_project/emb-final"
os.makedirs(save_dir, exist_ok=True)

train_csv_path = "/content/drive/MyDrive/pep_project/pepnn_train_dataset.csv"
test_csv_path  = "/content/drive/MyDrive/pep_project/pepnn_test_dataset.csv"

# =========================
# Load train and test datasets
# =========================
# Expected CSV columns:
# - 'Receptor Sequence' : protein (receptor) amino acid sequences
# - 'Sequence'          : peptide sequences
train_df = pd.read_csv(train_csv_path)
test_df  = pd.read_csv(test_csv_path)

train_protein_sequences = train_df['Receptor Sequence'].tolist()
train_pep_sequences     = train_df['Sequence'].tolist()
test_protein_sequences  = test_df['Receptor Sequence'].tolist()
test_pep_sequences      = test_df['Sequence'].tolist()

print(f"Train samples: {len(train_protein_sequences)} | Test samples: {len(test_protein_sequences)}")

# =========================
# Load ESM2 model
# =========================
# Using ESM2 (8M parameters) and extracting representations from layer 6
model, alphabet = pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
batch_converter = alphabet.get_batch_converter()

# Set model to evaluation mode (no dropout, no training behavior)
model.eval()

# Automatically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("✅ Loaded ESM2 8M model on", device)

# =========================
# Helper function for sequence embedding
# =========================
# This function:
# - Takes a list of amino acid sequences
# - Processes them in small batches
# - Runs them through ESM2
# - Extracts per-residue embeddings (L × D)
# - Removes BOS/EOS tokens
@torch.no_grad()
def embed_sequences(seq_list, batch_size=4):
    """Return list of per-residue embeddings (L, D) for each sequence."""
    all_embeds = []
    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding batches"):
        batch = seq_list[i:i+batch_size]

        # Prepare batch in (label, sequence) format required by ESM
        batch_data = [("seq"+str(j), s) for j, s in enumerate(batch)]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # Forward pass through the model
        out = model(batch_tokens, repr_layers=[6])
        reps = out["representations"][6]

        # Extract per-sequence embeddings without special tokens
        for j, seq in enumerate(batch_strs):
            emb = reps[j, 1:len(seq)+1].cpu()
            all_embeds.append(emb)

    return all_embeds

# =========================
# Utility function to save outputs in batches
# =========================
# Saves embeddings or tokens into multiple .pt files
# This avoids very large files and improves memory management
def save_batches(emb_list, prefix, kind, batch_size=1000):
    """Save batched embeddings or token lists to .pt files."""
    for i in range(0, len(emb_list), batch_size):
        batch = emb_list[i:i+batch_size]
        torch.save(batch, os.path.join(save_dir, f"{prefix}_{kind}_{i//batch_size:03d}.pt"))
    print(f"💾 Saved {len(emb_list)} items → {prefix}_{kind}_*.pt")

# =========================
# Generate embeddings for TRAIN set
# =========================
print("\n🚀 Embedding TRAIN proteins...")
train_prot_embs = embed_sequences(train_protein_sequences)
save_batches(train_prot_embs, "train", "prot_emb", batch_size=1000)

print("\n🚀 Embedding TRAIN peptides...")
train_pep_embs = embed_sequences(train_pep_sequences)
save_batches(train_pep_embs, "train", "pep_emb", batch_size=1000)

# =========================
# Generate embeddings for TEST set
# =========================
print("\n🚀 Embedding TEST proteins...")
test_prot_embs = embed_sequences(test_protein_sequences)
save_batches(test_prot_embs, "test", "prot_emb", batch_size=1000)

print("\n🚀 Embedding TEST peptides...")
test_pep_embs = embed_sequences(test_pep_sequences)
save_batches(test_pep_embs, "test", "pep_emb", batch_size=1000)

# =========================
# Peptide tokenization
# =========================
# Define amino acid vocabulary (20 standard amino acids)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"

# Create amino acid to index mapping
aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

# Convert peptide sequences into integer index tensors
def tokenize_sequences(seq_list):
    tokens = []
    for seq in seq_list:
        idxs = [aa_to_idx.get(aa, len(AA_LIST)) for aa in seq]
        tokens.append(torch.tensor(idxs, dtype=torch.long))
    return tokens

print("\n🔤 Tokenizing peptides...")
train_pep_tokens = tokenize_sequences(train_pep_sequences)
test_pep_tokens  = tokenize_sequences(test_pep_sequences)

# Save tokenized peptides
save_batches(train_pep_tokens, "train", "pep_token", batch_size=1000)
save_batches(test_pep_tokens, "test", "pep_token", batch_size=1000)

# =========================
# Final status message
# =========================
print("\n🎉 همه‌ی فایل‌ها ساخته شدند در پوشه‌ی:")
print(save_dir)
