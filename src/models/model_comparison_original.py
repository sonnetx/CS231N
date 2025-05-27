'''
This script is a baseline for comparing different image classification models
on the original ISIC 2019 dataset.
It has a set number of augmentation transforms and does NOT combine them.
This does NOT experiment on JPEG compression levels
'''

# Environment Setup
import os

os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging

# Standard Libraries
import io
import json
import random
import time

# Scientific & Visualization Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd

# PyTorch & Torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import transforms, datasets

# Hugging Face Transformers & Datasets
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)
from datasets import load_dataset, ClassLabel


# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Model Profiling & Vision Backbones
import timm
from thop import profile

# Local Application Imports
from utils.constants import HF_MODELS, SSL_MODEL, SIMCLR_BACKBONE, NUM_CLASSES, FILTERED_CLASSES, NUM_FILTERED_CLASSES
from utils.transforms import (
    JPEGCompressionTransform,
    GaussianBlurTransform,
    ColorQuantizationTransform,
)
from utils.util_classes import (
    ISICDataset,
    SimCLRForClassification,
    LossLoggerCallback,
)
from utils.util_methods import (
    env_path,
    compute_metrics,
    get_gpu_memory,
    freeze_backbone,
)

# GPU Memory Monitoring (optional)
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")

# Cache paths
os.environ["TRANSFORMERS_CACHE"] = os.getenv(
    "TRANSFORMERS_CACHE", "~/.cache/huggingface/transformers"
)
os.environ["HF_DATASETS_CACHE"] = os.getenv(
    "HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"
)
os.environ["HF_HOME"] = os.getenv("HF_HOME", "~/.cache/huggingface")

    
class CustomISICDataset(Dataset):
    def __init__(self, image_dir, label_file=None, transform=None, num_images=None, labels=["MEL", "NV"]):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # If you have a label file (e.g., CSV with image names and labels)
        if label_file:
            df = pd.read_csv(label_file)
            # Filter images based on provided labels
            self.images = [x + ".jpg" for x in df[(df['MEL'] == 1) | (df['NV'] == 1)]['image']]
            df['label'] = df['NV'].astype(int)  # Convert NV to 0 and MEL to 1
            self.image_labels = dict(zip(self.images, df['label']))
        else:
            raise ValueError("label_file must be provided to initialize image labels.")
        
        # Limit to num_images
        if num_images:
            # Make it 50% from both label classes
            mel_images = [x for x in self.images if self.image_labels[x] == 0]
            nv_images = [x for x in self.images if self.image_labels[x] == 1]
            self.images = mel_images[:num_images//2] + nv_images[:num_images//2]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[self.images[idx]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def main(num_train_images=1000, proportion_per_transform=0.2, resolution=224):
    models = [
        # {"name": "vit", "model_id": "google/vit-base-patch16-224", "type": "vit"},
        # {"name": "dinov2", "model_id": "facebook/dinov2-base", "type": "dinov2"},
        {"name": "simclr", "model_id": "resnet50", "type": "simclr"},
    ]

    results = {m["name"]: {} for m in models}
    results_linear_probe = {m["name"]: {} for m in models}

    data_dir = "/oak/stanford/groups/roxanad/ISIC_2019_Training_Input"
    label_file = "/oak/stanford/groups/roxanad/ISIC_2019_Training_GroundTruth.csv"

    dataset = CustomISICDataset(image_dir=data_dir, label_file=label_file, num_images=num_train_images)

    # Split into train and validation
    full_dataset = dataset.train_test_split(
        test_size=0.2, stratify_by_column="label", seed=42
    )

    train_dataset, val_dataset = full_dataset["train"], full_dataset["test"]
    
    degradation_transforms = [
        JPEGCompressionTransform(),
        GaussianBlurTransform(),
        ColorQuantizationTransform(),
    ]

    num_images = len(train_dataset)
    images_per_transform = int(num_images * proportion_per_transform)

    transformed_datasets = []
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    used_indices = []
    for i, transform in enumerate(degradation_transforms):
        subset_indices = indices[i * images_per_transform:(i + 1) * images_per_transform]
        used_indices.extend(subset_indices)
        subset = Subset(train_dataset, subset_indices)
        transform_compose = transforms.Compose([transform])
        
        for model_info in models:
            name, model_id, typ = (
                model_info["name"],
                model_info["model_id"],
                model_info["type"],
            )
            if typ == "vit":
                preprocessor = ViTFeatureExtractor.from_pretrained(model_id, size=resolution)
            elif typ == "dinov2":
                preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
            else:
                preprocessor = None

            transformed_ds = ISICDataset(subset, preprocessor, resolution, transform_compose, typ)
            transformed_datasets.append(transformed_ds)

    remaining_indices = np.setdiff1d(indices, used_indices)

    if len(remaining_indices) > 0:
        remaining_subset = Subset(train_dataset, remaining_indices)
        for model_info in models:
            name, model_id, typ = (
                model_info["name"],
                model_info["model_id"],
                model_info["type"],
            )
            if typ == "vit":
                preprocessor = ViTFeatureExtractor.from_pretrained(model_id, size=resolution)
            elif typ == "dinov2":
                preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
            else:
                preprocessor = None

            # No transform applied to remaining indices
            untransformed_ds = ISICDataset(remaining_subset, preprocessor, resolution, None, typ)
            transformed_datasets.append(untransformed_ds)

    train_ds = ConcatDataset(transformed_datasets)

    val_ds = ISICDataset(
        val_dataset,
        preprocessor,
        resolution,
        model_type=typ,
    )

    for model_info in models:
        name, model_id, typ = (
            model_info["name"],
            model_info["model_id"],
            model_info["type"],
        )
        if typ == "vit":
            preprocessor = ViTFeatureExtractor.from_pretrained(
                model_id, size=resolution
            )
        elif typ == "dinov2":
            preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
        else:
            preprocessor = None

        if typ == "vit":
            model = ViTForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == "dinov2":
            model = AutoModelForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == SSL_MODEL:
            # Load pretrained ResNet50 backbone
            backbone = timm.create_model(
                SIMCLR_BACKBONE,
                pretrained=True,
                num_classes=0  # Remove classification head
            )
            # Create SimCLR model with the backbone
            model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)
            # Freeze the backbone initially
            freeze_backbone(model, SSL_MODEL)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        try:
            dummy_input = torch.randn(1, 3, resolution, resolution).to(device)
            model.to(device)
            flops, _ = profile(model, inputs=(dummy_input,))
            flops /= 1e9
        except Exception as e:
            print(f"FLOP profiling failed: {e}")
            flops = -1

        train_args = TrainingArguments(
            output_dir=os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), f"{name}"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(env_path("LOG_DIR", "."), f"{name}"),
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,  # Only keep the best model
        )
        
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=lambda pred: compute_metrics(pred, name),
            callbacks=[
                LossLoggerCallback(
                    log_dir=env_path("LOG_DIR", "./logs"),
                    phase="finetune",
                    model_name=name,
                )
            ],
        )

        # ---- TRAINING PHASE ----
        start_time = time.time()
        peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1

        trainer.train()

        current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
        peak_memory = max(peak_memory, current_memory)

        eval_start_time = time.time()
        eval_results = trainer.evaluate()
        eval_time = time.time() - eval_start_time
        train_time = time.time() - start_time - eval_time

        model_dir = os.path.join(
            env_path("MODEL_DIR", "."), f"{name}"
        )
        os.makedirs(model_dir, exist_ok=True)

        if typ in HF_MODELS:
            model.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
        elif typ == SSL_MODEL:
            torch.save(
                model.state_dict(), os.path.join(model_dir, "pytorch_model.bin")
            )
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": SSL_MODEL,
                        "backbone": "resnet50",
                        "num_classes": NUM_FILTERED_CLASSES,
                    },
                    f,
                )

        results[name] = {
            "peak_memory_mb": peak_memory,
            "flops_giga": flops,
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "eval_metrics": eval_results,
        }

        print(
            f"[Finetune] {name}: {results[name]}"
        )

        # ---- LINEAR PROBE PHASE ----
        if typ == "vit":
            model = ViTForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == "dinov2":
            model = AutoModelForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == SSL_MODEL:
            backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
            model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)

        model.to(device)
        freeze_backbone(model, typ)

        linear_args = TrainingArguments(
            output_dir=os.path.join(
                env_path("TRAIN_OUTPUT_DIR", "."),
                f"{name}_linear_probe",
            ),
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(
                env_path("LOG_DIR", "."), f"{name}_linear_probe"
            ),
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,  # Only keep the best model
        )

        trainer = Trainer(
            model=model,
            args=linear_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=lambda pred: compute_metrics(pred, name),
            callbacks=[
                LossLoggerCallback(
                    log_dir=env_path("LOG_DIR", "./logs"),
                    phase="linear_probe",
                    model_name=name,
                )
            ],
        )

        start_time = time.time()
        peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1
        trainer.train()
        current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
        peak_memory = max(peak_memory, current_memory)

        eval_start_time = time.time()
        eval_results = trainer.evaluate()
        eval_time = time.time() - eval_start_time
        train_time = time.time() - start_time - eval_time

        model_dir = os.path.join(
            env_path("MODEL_DIR", "."), f"{name}_linear_probe"
        )
        os.makedirs(model_dir, exist_ok=True)

        if typ in HF_MODELS:
            model.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
        elif typ == SSL_MODEL:
            torch.save(
                model.state_dict(), os.path.join(model_dir, "pytorch_model.bin")
            )
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": SSL_MODEL,
                        "backbone": "resnet50",
                        "num_classes": NUM_FILTERED_CLASSES,
                    },
                    f,
                )

        results_linear_probe[name] = {
            "peak_memory_mb": peak_memory,
            "flops_giga": flops,
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "eval_metrics": eval_results,
        }

        print(
            f"[LinearProbe] {name}: {results_linear_probe[name]}"
        )

    with open(
        os.path.join(
            env_path("TRAIN_OUTPUT_DIR", "."), "results_metrics_finetune.json"
        ),
        "w",
    ) as f:
        json.dump(results, f, indent=4)

    with open(
        os.path.join(
            env_path("TRAIN_OUTPUT_DIR", "."), "results_metrics_linear_probe.json"
        ),
        "w",
    ) as f:
        json.dump(results_linear_probe, f, indent=4)


if __name__ == "__main__":
    main()
