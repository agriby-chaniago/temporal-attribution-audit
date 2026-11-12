"""Checkpoint model baru E1/E2. Training hanya melalui panggilan eksplisit notebook.

Tidak memakai atau menimpa artefak S3/S5. Impor modul tidak menjalankan training.
Identitas training terpisah dari parameter masker/plot agar checkpoint dapat dipakai ulang.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from audit_e0_e1 import AlignmentError, file_sha256


def spec_digest(spec: dict) -> str:
    return sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fold_spec(root: Path, cache_hash: str, records, train, test, arm: str,
              seed: int, fold: int, model_config: dict, training_config: dict,
              versions: dict) -> dict:
    return dict(
        format_version=1, cohort="newhandpd", cache_sha256=cache_hash,
        architecture=arm, seed=int(seed), fold=int(fold),
        model_config=model_config, training_config=training_config,
        split=dict(n_splits=5, shuffle=True, random_state=42,
                   train_indices=list(map(int, train)), test_indices=list(map(int, test)),
                   train_record_ids=records.record_id.iloc[train].tolist(),
                   test_record_ids=records.record_id.iloc[test].tolist()),
        sources={p: file_sha256(root / p) for p in (
            "src/model.py", "src/training.py", "src/preprocessing.py",
            "src/newhandpd.py", "src/e2_checkpoints.py")},
        versions=versions,
    )


def get_fold_model(root, spec, recordings, device, allow_training=True):
    """Muat checkpoint yang cocok atau latih satu model lengkap, lalu simpan atomik.

    Payload weights_only berisi tensor normalizer, split dan konfigurasi lengkap.
    Checkpoint lama tanpa format/metadata ini tidak dipulihkan secara spekulatif.
    """
    import torch
    from model import PDClassifier
    from preprocessing import Normalisasi
    from training import latih

    identity = spec_digest(spec)
    directory = Path(root) / "checkpoints/e2" / identity
    checkpoint = directory / "model.pt"
    metadata = directory / "metadata.json"
    config = dict(spec["model_config"], encoder=spec["architecture"])
    model = PDClassifier(**config).to(device)
    norm = Normalisasi()
    trained = False
    if checkpoint.exists():
        if not metadata.exists():
            raise AlignmentError(f"metadata checkpoint hilang: {directory}")
        meta = json.loads(metadata.read_text())
        if meta["spec"] != spec or meta["checkpoint_sha256"] != file_sha256(checkpoint):
            raise AlignmentError(f"spec/hash checkpoint tidak cocok: {directory}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["spec_json"] != json.dumps(spec, sort_keys=True):
            raise AlignmentError("spec payload checkpoint tidak cocok")
        model.load_state_dict(payload["state_dict"], strict=True)
        for name in ("tengah_", "skala_", "batas_"):
            setattr(norm, name, payload["normalizer"][name].numpy().copy())
        norm.batas_kuantil = float(payload["normalizer"]["batas_kuantil"])
        if any(getattr(norm, n).shape != (config["in_channels"],)
               or not np.isfinite(getattr(norm, n)).all()
               for n in ("tengah_", "skala_", "batas_")) or (norm.skala_ <= 0).any():
            raise AlignmentError("normalizer checkpoint tidak valid")
    else:
        if metadata.exists():
            raise AlignmentError(f"metadata ada tetapi checkpoint hilang: {directory}")
        if not allow_training:
            raise FileNotFoundError(f"checkpoint tidak tersedia dan training dinonaktifkan: {identity}")
        train = spec["split"]["train_indices"]
        norm.fit([recordings[i] for i in train])
        # Inisialisasi ulang setelah seed, persis prinsip pipeline sebelumnya.
        torch.manual_seed(spec["seed"])
        model = PDClassifier(**config).to(device)
        tc = spec["training_config"]
        history = latih(model, recordings, train, norm, epochs=tc["epochs"],
                        batch_size=tc["batch_size"], lr=tc["lr"],
                        seed=spec["seed"], device=device, verbose=True)
        payload = dict(
            spec_json=json.dumps(spec, sort_keys=True),
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            normalizer={**{n: torch.from_numpy(getattr(norm, n).copy())
                           for n in ("tengah_", "skala_", "batas_")},
                        "batas_kuantil": float(norm.batas_kuantil)},
        )
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "model.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
        meta = dict(spec=spec, checkpoint_sha256=file_sha256(checkpoint),
                    model_id=identity, loss_history=list(map(float, history)))
        temporary_meta = directory / "metadata.json.tmp"
        temporary_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(temporary_meta, metadata)
        trained = True
    model.eval()
    return model, norm, dict(model_id=identity, checkpoint=str(checkpoint),
                            checkpoint_sha256=file_sha256(checkpoint), trained_now=trained)


class FrozenPredictor:
    """Forward tanpa grad, panjang rekaman tetap; batch intervensi memakai T yang sama."""
    def __init__(self, model, device, length):
        self.model, self.device, self.length = model, device, int(length)
        self.model.eval()

    def __call__(self, array):
        import torch
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 3 or array.shape[1] != self.length:
            raise AlignmentError("panjang/bentuk input intervensi berubah")
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
            lengths = torch.full((len(array),), self.length, device=self.device, dtype=torch.long)
            logits, _ = self.model(x, lengths)
        return logits.cpu().numpy().astype(float)

    def original(self, array):
        import torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(array[None], dtype=np.float32)).to(self.device)
            lengths = torch.tensor([self.length], device=self.device)
            logit, alpha = self.model(x, lengths)
        n = self.length // self.model.patch_embed.patch_size
        return float(logit.item()), alpha[0, :n].cpu().numpy().astype(float)
