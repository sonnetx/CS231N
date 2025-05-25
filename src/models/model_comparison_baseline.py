'''
This script is a baseline for comparing different image classification models
at three different image compression levels, in comparison to the original.
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

# PyTorch & Torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import transforms

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
from constants import HF_MODELS, SSL_MODEL, SIMCLR_BACKBONE, NUM_CLASSES

# Constants for this script
FILTERED_CLASSES = ["0", "1"]  # Classes to use after filtering
NUM_FILTERED_CLASSES = len(FILTERED_CLASSES)  # Number of classes after filtering

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


def env_path(key, default):
    """Get environment variable or default value."""
    return os.environ.get(key, default)


class JPEGCompressionTransform:
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

class GaussianBlurTransform:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        if random.random() < self.p:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        return img

class ColorQuantizationTransform:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        if random.random() < self.p:
            num_colors = random.randint(16, 64)
            img = img.quantize(colors=num_colors, method=Image.Quantize.MAXCOVERAGE).convert("RGB")
        return img


class ISICDataset(Dataset):
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
            self.preprocessor = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert numpy.int64 to Python int if necessary
        if isinstance(idx, (np.integer, np.int64)):
            idx = int(idx)
            
        # Handle both direct dataset access and Subset access
        if hasattr(self.dataset, 'dataset'):
            # This is a Subset
            subset_idx = int(self.dataset.indices[idx])  # Convert the index from the indices array
            item = self.dataset.dataset[subset_idx]
        else:
            # This is a direct dataset
            item = self.dataset[idx]
            
        image = item["image"]
        label = item["label"]

        if self.resolution != 224:
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        if self.transform:
            image = self.transform(image)

        if self.jpeg_quality is not None:
            image = JPEGCompressionTransform(self.jpeg_quality)(image)

        if self.model_type in HF_MODELS:
            encoding = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)
        elif self.model_type == "simclr":
            pixel_values = self.preprocessor(image)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        label = torch.tensor(label, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label}


def compute_metrics(eval_pred, model_name):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    # For binary classification, use the probability of the positive class
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    # Use the probability of class 1 (positive class) for ROC AUC
    auc = roc_auc_score(labels, probs[:, 1])

    plot_dir = os.path.join(
        env_path("PLOT_DIR", "."), model_name
    )
    os.makedirs(plot_dir, exist_ok=True)

    conf_mat = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"{model_name}_conf_mat")
    plt.savefig(os.path.join(plot_dir, "conf_mat.png"), dpi=300, bbox_inches="tight")
    plt.close()

    unique, counts = np.unique(predictions, return_counts=True)
    class_breakdown = {str(k): int(v) for k, v in zip(unique, counts)}
    with open(os.path.join(plot_dir, "class_breakdown.json"), "w") as f:
        json.dump(class_breakdown, f)

    return {"accuracy": acc, "f1": f1, "auc": auc}


def get_gpu_memory(device_id=0):
    if not GPU_AVAILABLE:
        return -1
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2
    except:
        return -1


class SimCLRForClassification(nn.Module):
    def __init__(self, backbone, num_classes=NUM_FILTERED_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, pixel_values, labels=None):
        features = self.backbone(pixel_values)
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return (
            {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}
        )


def freeze_backbone(model, model_type):
    if model_type in HF_MODELS:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif model_type == "simclr":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


class LossLoggerCallback(TrainerCallback):
    """
    Logs each training step's loss and other metrics to a structured JSON Lines file.
    """

    def __init__(self, log_dir: str, phase: str, model_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, f"{model_name}_{phase}_log.jsonl"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with open(self.log_file, "a") as f:
            json.dump({"step": state.global_step, **logs}, f)
            f.write("\n")


def main(num_train_images=1000, proportion_per_transform=0.2, resolution=224):
    models = [
        {"name": "vit", "model_id": "google/vit-base-patch16-224", "type": "vit"},
        {"name": "dinov2", "model_id": "facebook/dinov2-base", "type": "dinov2"},
        # {"name": "simclr", "model_id": "resnet50", "type": "simclr"},
    ]

    results = {m["name"]: {} for m in models}
    results_linear_probe = {m["name"]: {} for m in models}

    dataset = load_dataset(
        "MKZuziak/ISIC_2019_224",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split=f"train[:{num_train_images}]",
    )

    # Filter to specified classes and cast labels - optimized version
    print("Filtering dataset for specified classes...")
    # Get indices of images with desired labels
    filtered_indices = [
        i for i, label in enumerate(dataset["label"])
        if str(label) in FILTERED_CLASSES  # Convert to string for comparison
    ]
    # Select only those indices
    dataset = dataset.select(filtered_indices)
    print(f"Number of images after filtering for classes {FILTERED_CLASSES}: {len(dataset)}")
    dataset = dataset.cast_column("label", ClassLabel(num_classes=NUM_FILTERED_CLASSES))

    # Get class counts and balance dataset - optimized version
    print("Balancing dataset...")
    # Get counts for each class
    class_counts = {label: 0 for label in FILTERED_CLASSES}
    for label in dataset["label"]:
        class_counts[str(label)] += 1  # Convert to string for dictionary key
    
    print(f"Class counts: {class_counts}")  # Debug print to verify counts
    
    # Calculate how many images to use per class
    min_class_size = min(class_counts.values())
    images_per_class = min(num_train_images // 2, min_class_size)
    
    # Sample indices for each class
    np.random.seed(42)
    balanced_indices = []
    for label in FILTERED_CLASSES:
        class_indices = [i for i, l in enumerate(dataset["label"]) if str(l) == label]  # Convert to string for comparison
        print(f"Found {len(class_indices)} images for class {label}")  # Debug print
        sampled_indices = np.random.choice(class_indices, images_per_class, replace=False)
        balanced_indices.extend(sampled_indices)
    
    np.random.shuffle(balanced_indices)
    balanced_dataset = dataset.select(balanced_indices)

    # Split into train and validation
    full_dataset = balanced_dataset.train_test_split(
        test_size=0.2, stratify_by_column="label", seed=42
    )

    train_dataset, val_dataset = full_dataset["train"], full_dataset["test"]
    
    degradation_transforms = [
        JPEGCompressionTransform(),
        GaussianBlurTransform(),
        ColorQuantizationTransform(),
    ]

    num_transforms = len(degradation_transforms)
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
        elif typ == "simclr":
            model = SimCLRForClassification(
                timm.create_model("resnet50", pretrained=True, num_classes=0), NUM_FILTERED_CLASSES
            )

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
