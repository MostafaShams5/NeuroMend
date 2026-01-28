import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

class Reasoner:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"[*] Initializing Reasoner with Qwen2.5-VL on {device}...")
        # Load 4-bit quantization if possible, otherwise standard fp16
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                quantization_config=bnb_config, 
                device_map="auto"
            )
        except ImportError:
            # Fallback for systems without bitsandbytes
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                torch_dtype=torch.float16
            ).to(self.device)
            
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    def analyze_cluster(self, image_paths):
        # We take the first 3 images of the cluster as representatives
        samples = [Image.open(p).convert("RGB") for p in image_paths[:3]]
        
        # Construct the VLM prompt
        text_prompt = (
            "Identify the common environmental conditions (weather, lighting, texture) "
            "in these images that might make object detection difficult. "
            "Output ONLY 3 keywords separated by commas."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": samples[0]},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages) # Helper usually needed for Qwen
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=30)
            
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Basic cleanup to just get the response
        return output_text.split("assistant")[-1].strip()

# Note: In real usage, Qwen's vision processing requires specific input handling 
# from their repo documentation. For Qwen 2.5, standard HF AutoProcessor usually works.
