"""
    Defect classifier training for Synthetic Inspector.

"""
import json
import argparse
from pathlib import Path 
from collections import Counter 

import torch 
import torch.nn as nn 
import torch.quantization 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models 
from PIL import Image 
from src.config import (
    DATA_IMAGE_DIR, DATA_ANNOTATION_DIR, MODEL_DIR,
    MODEL_FP32_PATH, MODEL_INT8_PATH, MODEL_ONNX_PATH, TRAINING_REPORT_PATH,
    IMAGE_SIZE, TRAIN_SPLIT, RANDOM_SEED, NUM_WORKERS, MAX_MODEL_MB,
    DEFECT_CLASSES, CLASS_TO_IDX, NUM_CLASSES,DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY
)

class DefectDataset(Dataset):
    def __init__ (self, image_paths: list, annotation_dir: Path, class_to_idx: dict, transform):
        self.image_paths = image_paths
        self.annotation_dir = annotation_dir
        self.class_to_idx = class_to_idx
        self.transform = transform 

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_dir / f"{image_path.stem}.json"

        with open(annotation_path) as f:
            label = self.class_to_idx[json.load(f)["class"]]
        
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def build_transforms() -> tuple:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


# Model 

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model 


# Evaluation
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = total = 0
    confusion = Counter()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                confusion[(l, p)] += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "samples": total,
        "confusion": dict(confusion),
    }    



def model_size_mb(model: nn.Module) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 * 1024)



# Quantization

def quantize_and_validate(fp32_model: nn.Module, val_loader: DataLoader, device: torch.device, class_names: list) -> tuple:
    print(f"[quant] FP32 size: {model_size_mb(fp32_model):.2f} MB")
    fp32_model_cpu = fp32_model.cpu().eval()

    quantized = torch.quantization.quantize_dynamic(
        fp32_model_cpu, {nn.Linear}, dtype=torch.qint8
    )

    print(f"[quant] INT8 size: {model_size_mb(quantized):.2f}MB")

    metrics = evaluate(quantized, val_loader, torch.device("cpu"))
    print(f"[quant] INT8 val accuracy: {metrics['accuracy']:.4f}")

    if model_size_mb(quantized) > MAX_MODEL_MB:
        raise ValueError(
            f"Quantized model {model_size_mb(quantized):.1f} MB exceeds {MAX_MODEL_MB} MB budget"
        )
    return quantized, metrics



# ONNX export for TensorRT (Jetson deployment path)
def export_onnx(fp32_model: nn.Module, num_classes: int, path: Path) -> None:
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        fp32_model.cpu().eval(),
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"[export] ONNX -> {path}")



# Main 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export (useful for quick iteration)")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True

    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[train] Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    #  discover classes from annotations
    annotation_files = list(DATA_ANNOTATION_DIR.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotations in {DATA_ANNOTATION_DIR}. Run generate.py first."
        )    

    class_names = sorted({
        json.loads(f.read_text())["class"]
        for f in annotation_files
    })

    class_to_idx = {n: i for i, n in enumerate(class_names)}
    print(f"[train] {len(class_names)} classes: {class_names}")

    # explicit per-split datasets 
    all_images = sorted(DATA_IMAGE_DIR.glob("*.png"))
    all_images = [p for p in all_images if (DATA_ANNOTATION_DIR / f"{p.stem}.json").exists()]
    if not all_images:
        raise ValueError("No valid image/annotation pairs.")
    
    rng = torch.Generator().manual_seed(RANDOM_SEED)
    perm = torch.randperm(len(all_images), generator=rng).tolist()
    n_train = int(len(all_images) * TRAIN_SPLIT)

    train_paths = [all_images[i] for i in perm[:n_train]]
    val_paths = [all_images[i] for i in perm[n_train:]]
    print(f"[train] {len(train_paths)} train | {len(val_paths)} val")

    train_tf, val_tf = build_transforms()
    train_set = DefectDataset(train_paths, DATA_ANNOTATION_DIR, class_to_idx, train_tf)
    val_set = DefectDataset(val_paths, DATA_ANNOTATION_DIR, class_to_idx, val_tf)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    
    # Model, loss, optimizer, scheduler, AMP
    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


    # training loop
    best_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        epoch_loss = running_loss / max(seen, 1)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[train] epoch {epoch+1:02d}/{args.epochs} | "
              f"loss {epoch_loss:.4f} | val_acc {metrics['accuracy']:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[train]   new best (val_acc {best_acc:.4f})")

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint. Check data/loss.")

    # save FP32, evaluate, quantize, RE-evaluate 
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_FP32_PATH)
    print(f"[train] Saved FP32 -> {MODEL_FP32_PATH} ({model_size_mb(model):.2f} MB)")

    fp32_metrics = evaluate(model, val_loader, device)

    quantized, int8_metrics = quantize_and_validate(
        model, val_loader, device, class_names
    )
    torch.save(quantized.state_dict(), MODEL_INT8_PATH)
    print(f"[train] Saved INT8 -> {MODEL_INT8_PATH}")

    # ONNX export from FP32 (TensorRT uses its own INT8)
    if not args.no_onnx:
        export_onnx(model, len(class_names), MODEL_ONNX_PATH)

    # report 
    report = {
        "class_names": class_names,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "best_val_accuracy_fp32": best_acc,
        "final_fp32_metrics": {k: v for k, v in fp32_metrics.items() if k != "confusion"},
        "int8_metrics": {k: v for k, v in int8_metrics.items() if k != "confusion"},
        "fp32_size_mb": round(model_size_mb(model), 2),
        "int8_size_mb": round(model_size_mb(quantized), 2),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "confusion_fp32": {f"{l}->{p}": c for (l, p), c in fp32_metrics["confusion"].items()},
    }
    TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[train] Report -> {TRAINING_REPORT_PATH}")
    print(f"[train] DONE. FP32 {fp32_metrics['accuracy']:.4f} | "
          f"INT8 {int8_metrics['accuracy']:.4f} | "
          f"{model_size_mb(quantized):.1f} MB")


if __name__ == "__main__":
    main()
