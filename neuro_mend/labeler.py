import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm
import json
import os

class AutoLabeler:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"[*] Initializing Labeler with Florence-2-Large on {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", 
            trust_remote_code=True
        ).to(self.device).eval() # Use .eval() to save memory
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", 
            trust_remote_code=True
        )

    def label_dataset(self, image_paths, target_class_name, output_dir):
        """
        Generates YOLO format labels for the images.
        """
        # Task: Caption to Phrase Grounding
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = f"{task_prompt} {target_class_name}"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"[*] Auto-labeling {len(image_paths)} images for class '{target_class_name}'...")
        
        for img_path in tqdm(image_paths):
            image = Image.open(img_path).convert("RGB")
            
            inputs = self.processor(text=text_input, images=image, return_tensors="pt").to(self.device, torch.float16)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            prediction = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            # Format: {'bboxes': [[x1, y1, x2, y2]], 'labels': ['pothole']}
            bboxes = prediction[task_prompt]['bboxes']
            
            # Convert to YOLO format (normalized center_x, center_y, w, h)
            yolo_lines = []
            for box in bboxes:
                x1, y1, x2, y2 = box
                dw = 1. / image.width
                dh = 1. / image.height
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + (w / 2)
                y_center = y1 + (h / 2)
                
                # Assume class ID 0 for simplicity (or map it)
                yolo_lines.append(f"0 {x_center*dw} {y_center*dh} {w*dw} {h*dh}")
            
            # Save .txt label file
            txt_name = os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt")
            with open(os.path.join(output_dir, txt_name), "w") as f:
                f.write("\n".join(yolo_lines))
