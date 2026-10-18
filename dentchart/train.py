"""Training loop with mixed precision, warm-up + cosine learning rate and resumable checkpoints."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import collate
from .evaluate import evaluate_map


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_detector(model, train_ds, val_ds, out_dir, epochs=12, batch_size=4, lr=0.01,
                   weight_decay=1e-4, warmup_iters=200, num_workers=2, device=None,
                   select_metric="AP50", resume=True, log=print):
    """Train a detector and keep the epoch with the best validation score.

    Writes out_dir/last.pt after every epoch (so a disconnected Colab session can resume)
    and out_dir/best.pt for the best epoch. Returns the training history (one dict per epoch).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or default_device()
    use_amp = device == "cuda"
    model.to(device)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=collate, pin_memory=use_amp)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    total_iters = max(1, epochs * len(loader))
    warmup = max(1, min(warmup_iters, total_iters // 5))

    def lr_factor(it):
        return min(1.0, (it + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(it, total_iters) / total_iters))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, history, best = 0, [], -1.0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, history, best = ck["epoch"], ck["history"], ck["best"]
        log(f"Resuming from epoch {start} (best {select_metric} so far {best:.3f})")

    for epoch in range(start, epochs):
        model.train()
        t0, running, steps = time.time(), 0.0, 0
        for images, targets in loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=use_amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            steps += 1

        scores = evaluate_map(model, val_ds, device=device, num_workers=num_workers)
        minutes = (time.time() - t0) / 60
        history.append({"epoch": epoch + 1, "loss": running / max(steps, 1), **scores, "minutes": minutes})
        if scores[select_metric] > best:
            best = scores[select_metric]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val": scores}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "history": history, "best": best}, last_path)
        log(f"epoch {epoch + 1:>2}/{epochs}  loss {history[-1]['loss']:.3f}  "
            f"val AP50 {scores['AP50']:.3f}  AP {scores['AP']:.3f}  ({minutes:.1f} min)")

    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    log(f"Best epoch: {ck['epoch']} (val {select_metric} {ck['val'][select_metric]:.3f})")
    return history
