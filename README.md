# NeuroMend: Auto-Healing for Computer Vision Models

NeuroMend is a tool that automatically finds weaknesses in your AI models and fixes them without human help. It identifies where your computer vision model is failing (like in bad weather or poor lighting), generates new training images to cover those gaps, and retrains the model to make it stronger.

## The Problem
When you train an object detection model (like YOLO), it often works well on clear, sunny days but fails when it rains, gets dark, or when the camera is blurry. Usually, fixing this requires a human to go out, collect more data in those specific conditions, and label it manually. This is slow and expensive.

## The Solution
NeuroMend automates this entire repair process in a closed loop. It works in four simple phases:

1. **Diagnosis:** It looks at your validation images and groups the failures together. For example, it might notice that your model keeps missing objects when it is foggy.
2. **Reasoning:** It uses a Vision-Language Model (VLM) to look at those failure groups and describe exactly what is wrong (e.g., "The image is dark and has motion blur").
3. **Synthesis:** It uses a high-speed image generator to create brand new, realistic training images that match those difficult conditions.
4. **Auto-Labeling:** It automatically draws boxes around the objects in these new images so they are ready for training immediately.

## Results
We tested NeuroMend on a Pothole Detection task where the model initially struggled with blurred and noisy images. You can see the full experiment in the [NeuroMend Notebook](neuro-mend.ipynb).

* **Baseline Model Accuracy:** 76.0% (mAP)
* **NeuroMend Model Accuracy:** 83.0% (mAP)
* **Improvement:** +7% absolute gain (+9.3% relative improvement)

By automatically generating 60 hard-case images and adding them to the training set, the model learned to see through the blur and noise significantly better.

## How It Works (Under the Hood)
NeuroMend combines several state-of-the-art technologies to achieve this:

* **Failure Detection:** Uses DINOv2 (by Meta) to understand image content and HDBSCAN to cluster similar failures.
* **Root Cause Analysis:** Uses Qwen2.5-VL to verbally describe the weather or lighting conditions causing the failure.
* **Data Generation:** Uses SDXL-Turbo (Stability AI) to generate corrective data in seconds.
* **Labeling:** Uses Florence-2 (Microsoft) to accurately detect and label objects in the new images.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MostafaShams5/NeuroMend.git
cd NeuroMend
```

### 2. Install Requirements

This project relies on very recent models, so you need specific versions of libraries like transformers and torch.

```bash
pip install -r requirements.txt
```

**Note:** You must have a GPU with at least 15GB of VRAM (like a T4, A10g, or RTX 3090/4090) because the pipeline loads large models sequentially.

## Usage

### Running the Full Pipeline

You can run the entire repair process using the main script.

1. Open `main_pipeline.py`.
2. Edit the CONFIG section at the top to point to your data:
```python
DATA_DIR = "./path/to/your/images"
OUTPUT_DIR = "./neuromend_output"
TARGET_CLASS = "pothole"
```

3. Run the script:
```bash
python main_pipeline.py
```

### Output

The script will create a folder (default: `neuromend_output`) containing:

* Folders for each failure cluster detected.
* Synthetically generated images that fix those failures.
* Text files with YOLO-formatted labels for the new images.

You can then copy these images and labels into your training folder and retrain your YOLO model to see the improvement.

## Project Structure

* `main_pipeline.py`: The master script that runs all 4 phases.
* `neuro_mend/diagnoser.py`: Finds the bad images.
* `neuro_mend/reasoner.py`: Explains why the images are bad.
* `neuro_mend/generator.py`: Creates new training data.
* `neuro_mend/labeler.py`: Labels the new data.
* `neuro_mend/utils.py`: Handles memory clearing to keep the GPU from crashing.

## Credits

**Author:** Mostafa Shams

This tool is built on top of open-source research from Meta (DINOv2), Alibaba Cloud (Qwen), Stability AI (SDXL), and Microsoft (Florence-2).
```
