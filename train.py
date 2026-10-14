"""Training entry point for BCM-UNet and the plain U-Net baseline.

Select the variant in config.py, then run:

    python train.py

Checkpoints, the CSV log and the periodic validation figures are written to
Config.OUTPUT_DIR.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config
from dataset import CrackDataset
from models import build_model
from utils.losses import BCEDiceLoss, JointEdgeSegLoss
from utils.visualization import visualize_batch_prediction


class AverageMeter:
    """Track the running mean of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(pred, target, threshold=0.5):
    """Pixel-wise precision, recall, IoU, mIoU and F1 for one batch."""
    with torch.no_grad():
        probs = torch.sigmoid(pred)
        preds = (probs > threshold).float().view(-1)
        target = target.view(-1)

        tp = (preds * target).sum()
        fp = (preds * (1 - target)).sum()
        fn = ((1 - preds) * target).sum()
        tn = ((1 - preds) * (1 - target)).sum()

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = tp / (tp + fp + fn + epsilon)
        miou = (iou + tn / (tn + fp + fn + epsilon)) / 2.0

        return precision.item(), recall.item(), iou.item(), miou.item(), f1.item()


def save_logs(log_path, epoch, train_metrics, val_metrics, lr):
    """Append one epoch of metrics to the CSV log."""
    if epoch == 0:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'lr', 'train_loss', 'train_iou', 'train_miou',
                'train_p', 'train_r', 'train_f1',
                'val_loss', 'val_iou', 'val_miou', 'val_p', 'val_r', 'val_f1',
            ])
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1, "{:.8f}".format(lr),
            "{:.4f}".format(train_metrics['loss'].avg),
            "{:.4f}".format(train_metrics['iou'].avg),
            "{:.4f}".format(train_metrics['miou'].avg),
            "{:.4f}".format(train_metrics['precision'].avg),
            "{:.4f}".format(train_metrics['recall'].avg),
            "{:.4f}".format(train_metrics['f1'].avg),
            "{:.4f}".format(val_metrics['loss'].avg),
            "{:.4f}".format(val_metrics['iou'].avg),
            "{:.4f}".format(val_metrics['miou'].avg),
            "{:.4f}".format(val_metrics['precision'].avg),
            "{:.4f}".format(val_metrics['recall'].avg),
            "{:.4f}".format(val_metrics['f1'].avg),
        ])


def create_dataloader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def train():
    Config.check_dataset()
    Config.setup_dirs()
    print('Using device: {}'.format(Config.DEVICE))
    torch.backends.cudnn.benchmark = True
    log_file = Config.LOG_FILE

    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
    ])

    full_dataset = CrackDataset(Config.IMAGE_DIR, Config.MASK_DIR, transform=transform)
    total = len(full_dataset)
    if total < 3:
        raise RuntimeError(
            'The dataset has {} samples, which is too few to form a 7:1:2 '
            'training/validation/test split.'.format(total))

    train_size = int(total * Config.TRAIN_SPLIT)
    val_size = int(total * Config.VAL_SPLIT)
    test_size = total - train_size - val_size

    # Rounding can starve a subset on small datasets; keep at least one sample
    # in validation and test so both remain usable.
    if val_size < 1:
        val_size = 1
        train_size = total - val_size - test_size
    if test_size < 1:
        test_size = 1
        train_size = total - val_size - test_size
    if train_size < 1:
        raise RuntimeError(
            'The dataset has {} samples, which is too few to form a 7:1:2 '
            'training/validation/test split.'.format(total))

    # Fixed seed for a reproducible split. The three subsets are disjoint.
    generator = torch.Generator().manual_seed(Config.SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    print('Split (7:1:2): {} training / {} validation / {} test samples'.format(
        train_size, val_size, test_size))

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)
    test_loader = create_dataloader(test_dataset, shuffle=False)

    # ==============================================================
    # Model: selected entirely by the switches in config.py
    # ==============================================================
    print('Run configuration: {}'.format(Config.exp_name))
    model = build_model(Config).to(Config.DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # The boundary module and the joint loss must agree: the edge branch is
    # only supervised when the module that predicts it exists.
    model_has_boundary = hasattr(model, 'boundary_extractor')
    if model_has_boundary != Config.USE_BOUNDARY_MODULE:
        raise RuntimeError(
            'Inconsistent configuration: USE_BOUNDARY_MODULE={} but the built '
            'model {} a boundary_extractor.'.format(
                Config.USE_BOUNDARY_MODULE,
                'has' if model_has_boundary else 'has no'))

    criterion_standard = BCEDiceLoss()
    criterion_joint = JointEdgeSegLoss(edge_weight=0.5).to(Config.DEVICE)
    scaler = GradScaler(device=str(Config.DEVICE))

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(Config.EPOCHS):
        metrics_keys = ['loss', 'iou', 'miou', 'precision', 'recall', 'f1']
        train_meters = {k: AverageMeter() for k in metrics_keys}
        val_meters = {k: AverageMeter() for k in metrics_keys}

        # -------- Train --------
        model.train()
        loop = tqdm(train_loader, desc='Epoch {}/{} [Train]'.format(epoch + 1, Config.EPOCHS))
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)

            with autocast(device_type=Config.DEVICE.type):
                outputs = model(images)

                # The boundary module returns (logits, edge_map); without it
                # the model returns the logits alone.
                if isinstance(outputs, tuple):
                    seg_pred, _edge_pred = outputs
                    loss = criterion_joint(outputs, masks)
                else:
                    seg_pred = outputs
                    loss = criterion_standard(seg_pred, masks)

                loss = loss / Config.ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if ((batch_idx + 1) % Config.ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            rec_loss = loss.item() * Config.ACCUMULATION_STEPS
            p, r, iou, miou, f1 = compute_metrics(seg_pred, masks)
            for k, v in zip(metrics_keys, [rec_loss, iou, miou, p, r, f1]):
                train_meters[k].update(v)

            loop.set_postfix({
                'L': "{:.4f}".format(train_meters['loss'].avg),
                'IoU': "{:.3f}".format(train_meters['iou'].avg),
                'mIoU': "{:.3f}".format(train_meters['miou'].avg),
                'P': "{:.3f}".format(train_meters['precision'].avg),
                'R': "{:.3f}".format(train_meters['recall'].avg),
                'F1': "{:.3f}".format(train_meters['f1'].avg),
            })

        # -------- Validation --------
        model.eval()
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc='[Val]', leave=False)
            for images, masks in val_loop:
                images = images.to(Config.DEVICE, non_blocking=True)
                masks = masks.to(Config.DEVICE, non_blocking=True)
                with autocast(device_type=Config.DEVICE.type):
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        seg_pred, _ = outputs
                        loss = criterion_joint(outputs, masks)
                    else:
                        seg_pred = outputs
                        loss = criterion_standard(seg_pred, masks)

                p, r, iou, miou, f1 = compute_metrics(seg_pred, masks)
                for k, v in zip(metrics_keys, [loss.item(), iou, miou, p, r, f1]):
                    val_meters[k].update(v)

                val_loop.set_postfix({
                    'L': "{:.4f}".format(val_meters['loss'].avg),
                    'IoU': "{:.3f}".format(val_meters['iou'].avg),
                    'mIoU': "{:.3f}".format(val_meters['miou'].avg),
                    'P': "{:.3f}".format(val_meters['precision'].avg),
                    'R': "{:.3f}".format(val_meters['recall'].avg),
                    'F1': "{:.3f}".format(val_meters['f1'].avg),
                })

        avg_train_loss = train_meters['loss'].avg
        avg_val_loss = val_meters['loss'].avg
        current_lr = optimizer.param_groups[0]['lr']

        print('\n' + '=' * 80)
        print('Epoch {} Summary:'.format(epoch + 1))
        print("  [Train] Loss: {:.4f} | IoU: {:.4f} | mIoU: {:.4f} | P: {:.4f} | R: {:.4f} | F1: {:.4f}".format(
            avg_train_loss, train_meters['iou'].avg, train_meters['miou'].avg,
            train_meters['precision'].avg, train_meters['recall'].avg, train_meters['f1'].avg))
        print("  [Val  ] Loss: {:.4f} | IoU: {:.4f} | mIoU: {:.4f} | P: {:.4f} | R: {:.4f} | F1: {:.4f}".format(
            avg_val_loss, val_meters['iou'].avg, val_meters['miou'].avg,
            val_meters['precision'].avg, val_meters['recall'].avg, val_meters['f1'].avg))
        print('  Learning Rate: {:.8f}'.format(current_lr))
        print('=' * 80)

        save_logs(log_file, epoch, train_meters, val_meters, current_lr)
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print('  Best model saved to {}'.format(Config.OUTPUT_DIR))
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.PATIENCE:
                print('\n  EARLY STOPPING TRIGGERED')
                break

        if (epoch + 1) % 10 == 0:
            try:
                model.eval()
                original_forward = model.forward

                def forward_seg_only(x, _f=original_forward):
                    out = _f(x)
                    return out[0] if isinstance(out, tuple) else out

                model.forward = forward_seg_only
                visualize_batch_prediction(model, val_loader, Config.DEVICE, epoch, Config.OUTPUT_DIR)
                model.forward = original_forward
            except Exception as exc:
                print('Visualization skipped: {}'.format(exc))

        torch.cuda.empty_cache()

    evaluate_on_test(model, test_loader, criterion_joint, criterion_standard)


def evaluate_on_test(model, test_loader, criterion_joint, criterion_standard):
    """Report the final metrics on the held-out test set."""
    metrics_keys = ['loss', 'iou', 'miou', 'precision', 'recall', 'f1']
    meters = {k: AverageMeter() for k in metrics_keys}

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='[Test]'):
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)

            with autocast(device_type=Config.DEVICE.type):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    seg_pred, _ = outputs
                    loss = criterion_joint(outputs, masks)
                else:
                    seg_pred = outputs
                    loss = criterion_standard(seg_pred, masks)

            p, r, iou, miou, f1 = compute_metrics(seg_pred, masks)
            for k, v in zip(metrics_keys, [loss.item(), iou, miou, p, r, f1]):
                meters[k].update(v)

    print('\n' + '=' * 80)
    print('Test set results (held out from training and model selection):')
    print("  Loss: {:.4f} | IoU: {:.4f} | mIoU: {:.4f} | Precision: {:.4f} | "
          "Recall: {:.4f} | F1: {:.4f}".format(
              meters['loss'].avg, meters['iou'].avg, meters['miou'].avg,
              meters['precision'].avg, meters['recall'].avg, meters['f1'].avg))
    print('=' * 80)

    with open(os.path.join(Config.OUTPUT_DIR, 'test_metrics.csv'), 'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'iou', 'miou', 'precision', 'recall', 'f1'])
        writer.writerow(["{:.4f}".format(meters[k].avg) for k in metrics_keys])
    return meters


if __name__ == '__main__':
    train()
