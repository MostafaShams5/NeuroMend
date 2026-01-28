import torch
import os
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from tqdm import tqdm

class Generator:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"[*] Initializing Generator with SDXL-Turbo on {device}...")
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def synthesize(self, source_images, prompt_modifiers, output_dir, count=50):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Base prompt structure
        # prompt_modifiers comes from the Reasoner (e.g., "Foggy, Low Light")
        full_prompt = f"high quality photo, realistic, {prompt_modifiers}"
        
        print(f"[*] Generating {count} images with prompt: '{full_prompt}'")
        
        generated_paths = []
        
        # Loop to generate data
        for i in tqdm(range(count)):
            # Pick a random source image to serve as the structural base
            source_img_path = source_images[i % len(source_images)]
            init_image = Image.open(source_img_path).convert("RGB").resize((512, 512))
            
            # Strength=0.6: Changes style/weather but keeps object shape
            image = self.pipe(
                prompt=full_prompt, 
                image=init_image, 
                strength=0.6, 
                guidance_scale=0.0, # Turbo needs 0.0 guidance
                num_inference_steps=2 # Turbo needs only 1-4 steps
            ).images[0]
            
            save_path = os.path.join(output_dir, f"syn_{i}.jpg")
            image.save(save_path)
            generated_paths.append(save_path)
            
        return generated_paths
