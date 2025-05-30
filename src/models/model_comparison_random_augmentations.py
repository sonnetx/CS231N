'''
This script compares the performance of different image classification models
It has more complicated and randomized data augmentation and experiments with
different JPEG compression levels.
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
from torch.utils.data import Dataset
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


class DegradationTransform:
    """
    Applies random JPEG compression and Gaussian blur to an image."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        if random.random() < self.p:
            quality = random.randint(10, 50)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        if random.random() < self.p:
            quality = random.randint(10, 50)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        if random.random() < self.p:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        if random.random() < self.p:
            num_colors = random.randint(16, 64)
            img = img.quantize(
                colors=num_colors, method=Image.Quantize.MAXCOVERAGE
            ).convert("RGB")
        return img


class JPEGCompressionTransform:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)


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


def compute_metrics(eval_pred, model_name, jpeg_quality):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    auc = roc_auc_score(labels, probs, multi_class="ovr")

    plot_dir = os.path.join(
        env_path("PLOT_DIR", "."), model_name, f"jpeg_{jpeg_quality}"
    )
    os.makedirs(plot_dir, exist_ok=True)

    conf_mat = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"{model_name}_jpeg{jpeg_quality}_conf_mat")
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
    def __init__(self, backbone, num_classes=8):
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

    def __init__(self, log_dir: str, phase: str, model_name: str, jpeg_quality: int):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, f"{model_name}_jpeg{jpeg_quality}_{phase}_log.jsonl"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with open(self.log_file, "a") as f:
            json.dump({"step": state.global_step, **logs}, f)
            f.write("\n")


def main(num_images=1000):
    models = [
        {"name": "vit", "model_id": "google/vit-base-patch16-224", "type": "vit"},
        {"name": "dinov2", "model_id": "facebook/dinov2-base", "type": "dinov2"},
        {"name": "simclr", "model_id": "resnet50", "type": "simclr"},
    ]
    jpeg_qualities = [90, 50, 20]
    resolution = 224

    results = {m["name"]: {} for m in models}
    results_linear_probe = {m["name"]: {} for m in models}

    dataset = load_dataset(
        "MKZuziak/ISIC_2019_224",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split=f"train[:{num_images}]",
    )

    dataset = dataset.cast_column("label", ClassLabel(num_classes=8))

    full_dataset = dataset["train"].train_test_split(
        test_size=0.2, stratify_by_column="label", seed=42
    )
    train_dataset, val_dataset = full_dataset["train"], full_dataset["test"]

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            DegradationTransform(p=0.5),
        ]
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

        train_ds = ISICDataset(train_dataset, preprocessor, resolution, transform, typ)
        if typ == "vit":
            model = ViTForImageClassification.from_pretrained(
                model_id, num_labels=8, ignore_mismatched_sizes=True
            )
        elif typ == "dinov2":
            model = AutoModelForImageClassification.from_pretrained(
                model_id, num_labels=8, ignore_mismatched_sizes=True
            )
        elif typ == "simclr":
            model = SimCLRForClassification(
                timm.create_model("resnet50", pretrained=True, num_classes=0), 8
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
        )

        for jpeg_quality in jpeg_qualities:
            val_ds = ISICDataset(
                val_dataset,
                preprocessor,
                resolution,
                model_type=typ,
                jpeg_quality=jpeg_quality,
            )
            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=lambda pred: compute_metrics(pred, name, jpeg_quality),
                callbacks=[
                    LossLoggerCallback(
                        log_dir=os.environ["LOG_DIR"],
                        phase="finetune",
                        model_name=name,
                        jpeg_quality=jpeg_quality,
                    )
                ],
            )

            # ---- TRAINING PHASE ----
            start_time = time.time()
            peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1

            if jpeg_quality == jpeg_qualities[0]:
                trainer.train()

            current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
            peak_memory = max(peak_memory, current_memory)

            eval_start_time = time.time()
            eval_results = trainer.evaluate()
            eval_time = time.time() - eval_start_time
            train_time = (
                time.time() - start_time - eval_time
                if jpeg_quality == jpeg_qualities[0]
                else 0
            )

            model_dir = os.path.join(
                env_path("MODEL_DIR", "."), f"{name}_jpeg{jpeg_quality}"
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
                            "num_classes": 8,
                        },
                        f,
                    )

            results[name][jpeg_quality] = {
                "peak_memory_mb": peak_memory,
                "flops_giga": flops,
                "train_time_seconds": train_time,
                "eval_time_seconds": eval_time,
                "eval_metrics": eval_results,
            }

            print(
                f"[Finetune] {name} @ JPEG {jpeg_quality}: {results[name][jpeg_quality]}"
            )

            # ---- LINEAR PROBE PHASE ----
            if typ == "vit":
                model = ViTForImageClassification.from_pretrained(
                    model_id, num_labels=8, ignore_mismatched_sizes=True
                )
            elif typ == "dinov2":
                model = AutoModelForImageClassification.from_pretrained(
                    model_id, num_labels=8, ignore_mismatched_sizes=True
                )
            elif typ == SSL_MODEL:
                backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
                model = SimCLRForClassification(backbone, 8)

            model.to(device)
            freeze_backbone(model, typ)

            linear_args = TrainingArguments(
                output_dir=os.path.join(
                    env_path("TRAIN_OUTPUT_DIR", "."),
                    f"{name}_jpeg{jpeg_quality}_linear_probe",
                ),
                num_train_epochs=1,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=os.path.join(
                    env_path("LOG_DIR", "."), f"{name}_jpeg{jpeg_quality}_linear_probe"
                ),
                logging_steps=1,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
            )

            trainer = Trainer(
                model=model,
                args=linear_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=lambda pred: compute_metrics(pred, name, jpeg_quality),
                callbacks=[
                    LossLoggerCallback(
                        log_dir=os.environ["LOG_DIR"],
                        phase="linear_probe",
                        model_name=name,
                        jpeg_quality=jpeg_quality,
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
                env_path("MODEL_DIR", "."), f"{name}_jpeg{jpeg_quality}_linear_probe"
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
                            "num_classes": 8,
                        },
                        f,
                    )

            results_linear_probe[name][jpeg_quality] = {
                "peak_memory_mb": peak_memory,
                "flops_giga": flops,
                "train_time_seconds": train_time,
                "eval_time_seconds": eval_time,
                "eval_metrics": eval_results,
            }

            print(
                f"[LinearProbe] {name} @ JPEG {jpeg_quality}: {results_linear_probe[name][jpeg_quality]}"
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
