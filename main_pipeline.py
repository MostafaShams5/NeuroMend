import os
import argparse
import sys
from neuro_mend import Diagnoser, Reasoner, Generator, AutoLabeler, flush_vram, load_image_batch

def main():
    # --- CLI CONFIGURATION ---
    parser = argparse.ArgumentParser(description="NeuroMend: Auto-Healing Pipeline for Vision Models")
    
    # Required arguments 
    parser.add_argument("--data_dir", type=str, required=True, help="Path to validation images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--target_class", type=str, required=True, help="Object class to fix (e.g., 'pothole')")
    
    # Tuning arguments
    parser.add_argument("--max_images", type=int, default=100, help="Max images to process")
    parser.add_argument("--gen_count", type=int, default=10, help="Synthetic images per cluster")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Min failure cluster size")
    
    args = parser.parse_args()

    # Input Validation
    if not os.path.exists(args.data_dir):
        print(f"[!] Error: Data directory '{args.data_dir}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- STEP 1: DIAGNOSIS ---
    all_images = load_image_batch(args.data_dir)
    
    if not all_images:
        print("[!] No images found in data directory.")
        sys.exit(1)
        
    # Apply limit
    all_images = all_images[:args.max_images]
    print(f"[*] Analyzing {len(all_images)} images...")

    diagnoser = Diagnoser()
    embeddings = diagnoser.get_embeddings(all_images)
    # Pass cluster size dynamically
    clusters = diagnoser.cluster_failures(embeddings, min_cluster_size=args.min_cluster_size)

    del diagnoser
    flush_vram()

    if not clusters:
        print("[-] No significant failure clusters found. Exiting.")
        return

    # --- STEP 2: REASONING ---
    reasoner = Reasoner()
    prompts = {}

    for c_id, indices in clusters.items():
        cluster_paths = [all_images[i] for i in indices]
        prompts[c_id] = reasoner.analyze_cluster(cluster_paths)
        print(f"Cluster {c_id}: {prompts[c_id]}")

    del reasoner
    flush_vram()

    # --- STEP 3: SYNTHESIS ---
    print("\n=== PHASE 3: SYNTHESIS ===")
    generator = Generator()
    synth_images = []

    for c_id, prompt in prompts.items():
        source_paths = [all_images[i] for i in clusters[c_id]]
        
        out_path = os.path.join(args.output_dir, c_id)
        # Pass generation count dynamically
        new_paths = generator.synthesize(source_paths, prompt, out_path, count=args.gen_count)
        synth_images.extend(new_paths)

    del generator
    flush_vram()

    # --- STEP 4: AUTO-LABELING ---
    print("\n=== PHASE 4: LABELING ===")
    labeler = AutoLabeler()
    label_out = os.path.join(args.output_dir, "labels")
    labeler.label_dataset(synth_images, args.target_class, label_out)

    del labeler
    flush_vram()

    print(f"\n Completed. Data available in {args.output_dir}")

if __name__ == "__main__":
    main()
