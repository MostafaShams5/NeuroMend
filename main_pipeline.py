import os
import torch
from neuro_mend import Diagnoser, Reasoner, Generator, AutoLabeler, flush_vram, load_image_batch

# --- CONFIG ---
DATA_DIR = "/kaggle/input/your-dataset/val_images" # Path to your validation set
OUTPUT_DIR = "/kaggle/working/neuromend_output"
TARGET_CLASS = "pothole" # Change this to your object

# --- STEP 1: DIAGNOSIS (CPU/GPU1) ---
print("\n=== PHASE 1: DIAGNOSIS ===")
# Get list of images (In a real run, filter this list to only include errors first)
all_images = load_image_batch(DATA_DIR)[:100] # Limit for demo

diagnoser = Diagnoser()
embeddings = diagnoser.get_embeddings(all_images)
clusters = diagnoser.cluster_failures(embeddings)

del diagnoser
flush_vram()

# --- STEP 2: REASONING (GPU1) ---
print("\n=== PHASE 2: REASONING ===")
reasoner = Reasoner()
prompts = {}

for c_id, indices in clusters.items():
    # Get file paths for this cluster
    cluster_paths = [all_images[i] for i in indices]
    prompts[c_id] = reasoner.analyze_cluster(cluster_paths)
    print(f"Cluster {c_id}: {prompts[c_id]}")

del reasoner
flush_vram()

# --- STEP 3: SYNTHESIS (GPU1) ---
print("\n=== PHASE 3: SYNTHESIS ===")
generator = Generator()
synth_images = []

for c_id, prompt in prompts.items():
    # Use images from the cluster as base for Img2Img
    source_paths = [all_images[i] for i in clusters[c_id]]
    
    out_path = os.path.join(OUTPUT_DIR, c_id)
    new_paths = generator.synthesize(source_paths, prompt, out_path, count=10) # 10 for demo
    synth_images.extend(new_paths)

del generator
flush_vram()

# --- STEP 4: AUTO-LABELING (GPU1) ---
print("\n=== PHASE 4: LABELING ===")
labeler = AutoLabeler()
label_out = os.path.join(OUTPUT_DIR, "labels")
labeler.label_dataset(synth_images, TARGET_CLASS, label_out)

del labeler
flush_vram()

print(f"\n[SUCCESS] Pipeline Complete. Data available in {OUTPUT_DIR}")
