# -*- coding: utf-8 -*-
"""Join the transcripts of multi-file interviews.

A multi-file interview is split into one recipe item per file (`recipe.split_into_parts`),
so each part is preprocessed, analysed and diarized on its own: a new file usually means a
new room or recording device. Part transcripts are written to `<output_base_folder>/_parts/`.

`merge_part_transcripts` joins them into one `Interview_<id>.csv` in the main transcripts
folder, with extra columns:

- `Part`: part number (1, 2, ...)
- `Audio`: FLAC/compact-audio stem of that part (e.g. `Interview_20_part2`); timestamps
  are relative to that part's audio
- `PartSpeaker`: the speaker label diarization gave within that part

`Speaker` is harmonised across parts: diarization labels are arbitrary per part, so each
part's speakers are matched to part 1's by comparing pyannote voice embeddings.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qualivault.recipe import load_recipe, part_output_stem

logger = logging.getLogger("QualiVault")

PARTS_SUBFOLDER = "_parts"
EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"


def parts_dir(output_dir: Path) -> Path:
    return Path(output_dir) / PARTS_SUBFOLDER


def _group_parts(recipe: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in recipe:
        if it.get("part") and it.get("interview_id"):
            groups[str(it["interview_id"])].append(it)
    for iid in groups:
        groups[iid].sort(key=lambda it: int(it["part"]))
    return dict(groups)


class _SpeakerEmbedder:
    """Average pyannote speaker embedding of a speaker's longest segments in one part."""

    def __init__(self, hf_token: Optional[str], device: str, cache_dir: Optional[Path], preprocessing: Dict[str, Any]):
        import torch
        from pyannote.audio import Inference, Model

        kwargs: Dict[str, Any] = {"token": hf_token} if hf_token else {}
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
        model = Model.from_pretrained(EMBEDDING_MODEL_ID, **kwargs)
        self.inference = Inference(model, window="whole")
        self.inference.to(torch.device(device))
        self.torch = torch
        self.trim = bool(preprocessing.get("trim_silence", True))
        self.trim_top_db = max(10.0, float(preprocessing.get("trim_silence_top_db", 30.0)))

    def load(self, flac_path: Path) -> np.ndarray:
        import librosa

        from qualivault.audio import trim_silence

        y, _ = librosa.load(flac_path, sr=16000, mono=True)
        if self.trim:
            # Transcript timestamps are relative to the trimmed audio (see core.process_item).
            try:
                y, _ = trim_silence(y, top_db=self.trim_top_db)
            except Exception:
                pass
        return y

    def speaker_centroids(self, y: np.ndarray, df: pd.DataFrame, max_seconds: float = 120.0) -> Dict[str, np.ndarray]:
        centroids: Dict[str, np.ndarray] = {}
        for speaker, rows in df.groupby("Speaker"):
            rows = rows.assign(dur=rows["End"] - rows["Start"]).sort_values("dur", ascending=False)
            vecs, weights, total = [], [], 0.0
            for _, r in rows.iterrows():
                if r["dur"] < 1.0 or total >= max_seconds:
                    break
                a, b = int(r["Start"] * 16000), int(r["End"] * 16000)
                chunk = y[a:b]
                if len(chunk) < 16000:
                    continue
                wav = self.torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None, :]
                emb = np.asarray(self.inference({"waveform": wav, "sample_rate": 16000})).reshape(-1)
                if not np.all(np.isfinite(emb)):
                    continue
                vecs.append(emb / (np.linalg.norm(emb) + 1e-9))
                weights.append(r["dur"])
                total += r["dur"]
            if vecs:
                c = np.average(np.stack(vecs), axis=0, weights=weights)
                centroids[str(speaker)] = c / (np.linalg.norm(c) + 1e-9)
        return centroids


def _match_labels(ref: Dict[str, np.ndarray], cur: Dict[str, np.ndarray], used_labels: List[str]) -> Dict[str, tuple]:
    """Map each label in `cur` to a label in `ref` (Hungarian on cosine similarity).

    Returns {cur_label: (new_label, similarity or None)}. Speakers with no counterpart get a new label.
    """
    from scipy.optimize import linear_sum_assignment

    mapping: Dict[str, tuple] = {}
    ref_labels, cur_labels = list(ref), list(cur)
    if ref_labels and cur_labels:
        sim = np.array([[float(np.dot(cur[c], ref[r])) for r in ref_labels] for c in cur_labels])
        rows, cols = linear_sum_assignment(-sim)
        for i, j in zip(rows, cols):
            mapping[cur_labels[i]] = (ref_labels[j], round(float(sim[i, j]), 3))
    taken = set(used_labels)
    n = 0
    for c in cur_labels:
        if c not in mapping:
            while f"SPEAKER_{n:02d}" in taken:
                n += 1
            mapping[c] = (f"SPEAKER_{n:02d}", None)
            taken.add(f"SPEAKER_{n:02d}")
    return mapping


def merge_part_transcripts(
    *,
    project_root: Path,
    config: Dict[str, Any],
    recipe_path: Optional[Path] = None,
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
    match_speakers: bool = True,
    force: bool = False,
) -> List[Path]:
    """Join part transcripts into one CSV per interview. Returns the CSVs written.

    An interview is joined only when every part listed in the recipe has a transcript.
    Existing joined CSVs are rewritten only when a part CSV is newer, or with `force=True`.
    """
    project_root = Path(project_root)
    recipe = load_recipe(Path(recipe_path) if recipe_path else project_root / "processing_recipe.yaml")
    groups = _group_parts(recipe)
    if not groups:
        return []

    paths_cfg = config.get("paths", {})
    output_dir = (project_root / Path(paths_cfg["output_base_folder"]).expanduser()).resolve()
    flac_dir = (project_root / Path(paths_cfg["flac_output_folder"]).expanduser()).resolve()
    models_dir = paths_cfg.get("models_folder")
    cache_dir = (project_root / models_dir).resolve() if models_dir else None
    preprocessing = config.get("transcription", {}).get("preprocessing", {})

    embedder = None
    written: List[Path] = []
    for iid, items in groups.items():
        part_csvs = [parts_dir(output_dir) / f"{part_output_stem(iid, int(it['part']))}.csv" for it in items]
        missing = [p.name for p in part_csvs if not p.exists()]
        if missing:
            logger.info(f"⏭️  Interview_{iid}: waiting for parts {missing}")
            continue
        out_csv = output_dir / f"Interview_{iid}.csv"
        if out_csv.exists() and not force and out_csv.stat().st_mtime >= max(p.stat().st_mtime for p in part_csvs):
            continue

        frames = []
        ref: Dict[str, np.ndarray] = {}
        for it, csv in zip(items, part_csvs):
            n = int(it["part"])
            stem = part_output_stem(iid, n)
            df = pd.read_csv(csv)
            df["Text"] = df["Text"].fillna("")
            df["PartSpeaker"] = df["Speaker"].astype(str)
            df["Part"] = n
            df["Audio"] = stem

            if match_speakers and len(items) > 1 and not df.empty:
                try:
                    if embedder is None:
                        from qualivault.core import get_device

                        embedder = _SpeakerEmbedder(hf_token, device or get_device(), cache_dir, preprocessing)
                    cents = embedder.speaker_centroids(embedder.load(flac_dir / f"{stem}.flac"), df)
                    if not ref:
                        ref = cents  # part 1 defines the labels
                        logger.info(f"🎙️  Interview_{iid} part {n}: reference speakers {sorted(cents)}")
                    else:
                        used = sorted(set(ref) | set(df["Speaker"].astype(str)))
                        mapping = _match_labels(ref, cents, used)
                        # Speakers too short to embed keep a part-specific label rather than a guessed one.
                        df["Speaker"] = df["PartSpeaker"].map(lambda s: mapping.get(s, (f"{s} (del {n})", None))[0])
                        logger.info(
                            f"🎙️  Interview_{iid} part {n}: "
                            + ", ".join(f"{k}→{v[0]} (sim {v[1]})" for k, v in sorted(mapping.items()))
                        )
                        for k, (label, _) in mapping.items():
                            ref.setdefault(label, cents[k])
                except Exception as e:
                    logger.warning(f"⚠️ Interview_{iid} part {n}: speaker matching skipped ({e}); labels kept per part")
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        cols = ["Start", "End", "Speaker", "Text", "Part", "Audio", "PartSpeaker"]
        merged = merged[cols + [c for c in merged.columns if c not in cols]]
        merged.to_csv(out_csv, index=False)
        written.append(out_csv)
        logger.info(f"🧩 Joined {len(items)} parts -> {out_csv.name} ({len(merged)} segments)")
    return written
