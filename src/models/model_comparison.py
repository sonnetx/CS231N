'''
This script fine-tunes ViT, DINOv2, and SimCLR models on the ISIC 2019 dataset with various resolutions.
It includes data augmentation with degradation (compress-decompress, blur, color quantization) to train robust models.
DINOv2 is loaded from Hugging Face.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
from transformers import AutoModelForImageClassification, AutoImageProcessor  # For DINOv2
from datasets import load_dataset, ClassLabel
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import timm  # For SimCLR's ResNet backbone
import random
import io

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")
from thop import profile

# Custom transform for degradation augmentations
class DegradationTransform:
    def __init__(self, p=0.5):
        self.p = p  # Probability of applying each degradation

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        # JPEG compression
        if random.random() < self.p:
            quality = random.randint(10, 50)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        
        # Compress-decompress cycle (JPEG)
        if random.random() < self.p:
            quality = random.randint(10, 50)  # Random JPEG quality
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        
        # Gaussian blur
        if random.random() < self.p:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        
        # Color quantization
        if random.random() < self.p:
            num_colors = random.randint(16, 64)
            img = img.quantize(colors=num_colors, method=Image.Quantize.MAXCOVERAGE).convert('RGB')
        
        return img
    
class JPEGCompressionTransform:
    """Applies JPEG compression at a specified quality level for validation."""

    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

# Custom Dataset for ISIC 2019 with Downsampling and Model-Specific Preprocessing
class ISICDataset(Dataset):
    """
    Custom dataset for ISIC 2019 with model-specific preprocessing and optional
    JPEG compression.
    """

    def __init__(
        self,
        dataset,
        preprocessor=None,
        resolution=224,
        transform=None,
        model_type="vit",
        jpeg_quality=None,
    ):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.resolution = resolution
        self.transform = transform
        self.model_type = model_type
        self.jpeg_quality = jpeg_quality

        if model_type == "simclr":
            self.preprocessor = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.resolution != 224:
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        if self.transform:
            image = self.transform(image)

        if self.jpeg_quality is not None:
            image = JPEGCompressionTransform(self.jpeg_quality)(image)

        if self.model_type in ["vit", "dinov2"]:
            encoding = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)
        elif self.model_type == "simclr":
            pixel_values = self.preprocessor(image)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        label = torch.tensor(label, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label}

# Compute metrics for evaluation
def compute_metrics(eval_pred, model_name, jpeg_quality):
    """Computes evaluation metrics and saves confusion matrix and class breakdown."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    auc = roc_auc_score(labels, logits, multi_class="ovr")

    # Save confusion matrix
    conf_mat = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"{model_name}_jpeg{jpeg_quality}_conf_mat")
    plt.savefig(
        f"{model_name}_jpeg{jpeg_quality}_conf_mat.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save class breakdown
    unique, counts = np.unique(predictions, return_counts=True)
    class_breakdown = dict(zip(unique, counts))
    with open(f"{model_name}_jpeg{jpeg_quality}_class_breakdown.json", "w") as f:
        json.dump(class_breakdown, f)

    return {"accuracy": acc, "f1": f1, "auc": auc}

# Measure GPU memory usage
def get_gpu_memory(device_id=0):
    if not GPU_AVAILABLE:
        return -1
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2  # MB
    except:
        return -1
    
# Wrapper for SimCLR to match Trainer API
class SimCLRForClassification(nn.Module):
    def __init__(self, backbone, num_classes=8):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)  # ResNet-50 output: 2048

    def forward(self, pixel_values, labels=None):
        features = self.backbone(pixel_values).pooler_output  # Get pooled output
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'logits': logits, 'loss': loss} if loss is not None else {'logits': logits}

def freeze_backbone(model, model_type):
    """Freezes the backbone of the model, leaving only the classifier trainable."""
    if model_type in ["vit", "dinov2"]:
        for name, param in model.named_parameters():
            if "classifier" not in name:  # Train only the classifier head
                param.requires_grad = False
    elif model_type == "simclr":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")





# Main function for fine-tuning
def main():
    """Fine-tunes models and evaluates performance on JPEG-compressed validation
    images.
    """
    models = [
        {"name": "vit", "model_id": "google/vit-base-patch16-224", "type": "vit"},
        {"name": "dinov2", "model_id": "facebook/dinov2-base", "type": "dinov2"},
        {"name": "simclr", "model_id": "resnet50", "type": "simclr"},
    ]
    jpeg_qualities = [90, 50, 20]  # High, medium, low quality
    resolution = 224  # Fixed resolution

    results = {model["name"]: {} for model in models}
    results_linear_probe = {model["name"]: {} for model in models}

    # Load dataset
    dataset = load_dataset("MKZuziak/ISIC_2019_224", cache_dir=os.environ["HF_DATASETS_CACHE"])
    dataset = dataset.cast_column("label", ClassLabel(num_classes=9))
    full_dataset = dataset["train"].train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

    train_dataset, val_dataset = full_dataset["train"], full_dataset["test"]

    # Define training augmentations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        DegradationTransform(p=0.5),
    ])

    for model_info in models:
        model_name = model_info["name"]
        model_id = model_info["model_id"]
        model_type = model_info["type"]
        print(f"\nFine-tuning model: {model_name}")

        # Load preprocessor
        if model_type == "vit":
            preprocessor = ViTFeatureExtractor.from_pretrained(model_id, size=resolution)
        elif model_type == "dinov2":
            preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
        elif model_type == "simclr":
            preprocessor = None
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Create training dataset
        train_ds = ISICDataset(
            train_dataset,
            preprocessor,
            resolution,
            transform,
            model_type,
            jpeg_quality=None,
        )

        # Load model
        if model_type == "vit":
            model = ViTForImageClassification.from_pretrained(
                model_id, num_labels=8, ignore_mismatched_sizes=True
            )
        elif model_type == "dinov2":
            model = AutoModelForImageClassification.from_pretrained(
                model_id, num_labels=8, ignore_mismatched_sizes=True
            )
        elif model_type == "simclr":
            backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
            model = SimCLRForClassification(backbone, num_classes=8)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Estimate FLOPs
        input_tensor = torch.randn(1, 3, resolution, resolution).to(device)
        try:
            flops, _ = profile(model, inputs=(input_tensor,))
            flops = flops / 1e9  # Convert to GFLOPs
        except Exception:
            flops = -1

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), f"{name}"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(env_path("LOG_DIR", "."), f"{name}"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        for jpeg_quality in jpeg_qualities:
            print(f"JPEG Quality: {jpeg_quality}")

            # Create validation dataset with JPEG compression
            val_ds = ISICDataset(
                val_dataset,
                preprocessor,
                resolution,
                model_type=model_type,
                jpeg_quality=jpeg_quality,
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=lambda pred: compute_metrics(
                    pred, model_name, jpeg_quality
                ),
            )

            start_time = time.time()
            peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1

            # Train only for the first JPEG quality
            if jpeg_quality == jpeg_qualities[0]:
                trainer.train()

            current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
            peak_memory = max(peak_memory, current_memory)
            # Evaluate
            eval_start_time = time.time()
            eval_results = trainer.evaluate()
            eval_time = time.time() - eval_start_time

            train_time = (
                time.time() - start_time - eval_time
                if jpeg_quality == jpeg_qualities[0]
                else 0
            )

            # Save model and preprocessor
            model.save_pretrained(f"./finetuned_{model_name}_jpeg{jpeg_quality}")
            if model_type in ["vit", "dinov2"]:
                preprocessor.save_pretrained(
                    f"./finetuned_{model_name}_jpeg{jpeg_quality}"
                )

            # Store results
            results[model_name][jpeg_quality] = {
                "peak_memory_mb": peak_memory,
                "flops_giga": flops,
                "train_time_seconds": train_time,
                "eval_time_seconds": eval_time,
                "eval_metrics": eval_results,
            }
            print(
                f"Results for {model_name} at JPEG quality {jpeg_quality}: "
                f"{results[model_name][jpeg_quality]}"
            )

            # Linear probing
            print(f"\nLinear probing at JPEG quality: {jpeg_quality}")

            # Reload model for linear probing
            if model_type == "vit":
                model = ViTForImageClassification.from_pretrained(
                    model_id, num_labels=8, ignore_mismatched_sizes=True
                )
            elif model_type == "dinov2":
                model = AutoModelForImageClassification.from_pretrained(
                    model_id, num_labels=8, ignore_mismatched_sizes=True
                )
            elif model_type == "simclr":
                backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
                # Optional: Load true SimCLR weights if available
                # backbone.load_state_dict(torch.load('path_to_simclr_weights.pth'))
                model = SimCLRForClassification(backbone, num_classes=8)

            model.to(device)
            freeze_backbone(model, model_type)

            # Linear probing training arguments
            linear_probe_args = TrainingArguments(
                output_dir=f"./results_{model_name}_jpeg{jpeg_quality}_linear_probe",
                num_train_epochs=1,  # Fewer epochs for linear probing
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=f"./logs_{model_name}_jpeg{jpeg_quality}_linear_probe",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
            )

            # Initialize trainer for linear probing
            trainer = Trainer(
                model=model,
                args=linear_probe_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=lambda pred: compute_metrics(
                    pred, model_name, jpeg_quality, mode="linear_probe"
                ),
            )

            # Linear probe
            start_time = time.time()
            peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1
            trainer.train()
            current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
            peak_memory = max(peak_memory, current_memory)

            # Evaluate linear probing
            eval_start_time = time.time()
            eval_results = trainer.evaluate()
            eval_time = time.time() - eval_start_time
            train_time = time.time() - start_time - eval_time

            # Save linear probe model
            model.save_pretrained(
                f"./finetuned_{model_name}_jpeg{jpeg_quality}_linear_probe"
            )
            if model_type in ["vit", "dinov2"]:
                preprocessor.save_pretrained(
                    f"./finetuned_{model_name}_jpeg{jpeg_quality}_linear_probe"
                )

            # Store linear probing results
            results_linear_probe[model_name][jpeg_quality] = {
                "peak_memory_mb": peak_memory,
                "flops_giga": flops,
                "train_time_seconds": train_time,
                "eval_time_seconds": eval_time,
                "eval_metrics": eval_results,
            }
            print(
                f"Linear probing results for {model_name} at JPEG quality "
                f"{jpeg_quality}: {results_linear_probe[model_name][jpeg_quality]}"
            )

    # Save results
    with open("results_metrics_finetune.json", "w") as f:
        json.dump(results, f, indent=4)
    with open("results_metrics_linear_probe.json", "w") as f:
        json.dump(results_linear_probe, f, indent=4)

if __name__ == '__main__':
    main()