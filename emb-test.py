# =========================
# Install required packages (already assumed installed in Colab)
# =========================


# =========================
# Imports
# =========================
import torch, os, pandas as pd
from tqdm import tqdm
from esm import pretrained

# =========================
# Output directory setup
# This directory will store all generated embeddings and token files
# =========================
save_dir = "/content/drive/MyDrive/pep_project/emb-final/test"
os.makedirs(save_dir, exist_ok=True)

# Path to test CSV file
test_csv_path  = "/content/drive/MyDrive/pep_project/test.csv"

# =========================
# Load test dataset
# =========================
# The CSV is expected to contain:
# - 'Receptor Sequence' column for protein sequences
# - 'Binder' column for peptide sequences
test_df  = pd.read_csv(test_csv_path)

test_protein_sequences  = test_df['Receptor Sequence'].tolist()
test_pep_sequences      = test_df['Binder'].tolist()

print(f"Test samples: {len(test_protein_sequences)}")

# =========================
# Load ESM2 model
# =========================
# Using ESM2 8M parameter model (layer 6 representations)
model, alphabet = pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
batch_converter = alphabet.get_batch_converter()
model.eval()  # Set model to evaluation mode (no dropout)

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("✅ Loaded ESM2 8M model on", device)

# =========================
# Helper function: sequence embedding
# =========================
# This function:
# - Processes sequences in batches
# - Extracts per-residue embeddings from layer 6
# - Removes BOS/EOS tokens
# - Returns a list of tensors (variable length per sequence)
@torch.no_grad()
def embed_sequences(seq_list, batch_size=4):
    all_embeds = []

    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding batches"):
        batch = seq_list[i:i+batch_size]

        # Prepare batch in ESM expected format: (label, sequence)
        batch_data = [("seq"+str(j), s) for j, s in enumerate(batch)]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # Forward pass through ESM
        out = model(batch_tokens, repr_layers=[6])
        reps = out["representations"][6]

        # Extract per-sequence embeddings (excluding special tokens)
        for j, seq in enumerate(batch_strs):
            emb = reps[j, 1:len(seq)+1].cpu()
            all_embeds.append(emb)

    return all_embeds

# =========================
# Peptide tokenization setup
# =========================
# Amino acid vocabulary (20 standard amino acids)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"

# Mapping amino acids to indices
aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

# Padding index for unknown or non-standard amino acids
PAD_IDX = len(AA_LIST)

# =========================
# Tokenization function for peptide sequences
# =========================
# Converts peptide strings into integer tensors
# Each amino acid is mapped to its index
def tokenize_sequences(seq_list):
    tokens = []
    for seq in seq_list:
        idxs = [aa_to_idx.get(aa, PAD_IDX) for aa in seq]
        tokens.append(torch.tensor(idxs, dtype=torch.long))
    return tokens

# =========================
# Batch-saving utility
# =========================
# Saves data in chunks (default: 1000 items per file)
# This prevents very large .pt files and improves memory handling
def save_batches(emb_list, prefix, kind, batch_size=1000):
    for i in range(0, len(emb_list), batch_size):
        batch = emb_list[i:i+batch_size]
        torch.save(
            batch,
            os.path.join(save_dir, f"{prefix}_{kind}_{i//batch_size:03d}.pt")
        )
    print(f"💾 Saved {len(emb_list)} items → {prefix}_{kind}_*.pt")

# =========================
# Generate embeddings for TEST proteins
# =========================
print("\n🚀 Embedding TEST proteins...")
test_prot_embs = embed_sequences(test_protein_sequences)
save_batches(test_prot_embs, "test", "prot_emb")

# =========================
# Generate embeddings for TEST peptides
# =========================
print("\n🚀 Embedding TEST peptides...")
test_pep_embs = embed_sequences(test_pep_sequences)
save_batches(test_pep_embs, "test", "pep_emb")

# =========================
# Tokenize TEST peptides
# =========================
print("\n🔤 Tokenizing TEST peptides...")
test_pep_tokens = tokenize_sequences(test_pep_sequences)
save_batches(test_pep_tokens, "test", "pep_token")

# =========================
# Final status
# =========================
print("\n🎉 All test files successfully generated at:")
print(save_dir)
