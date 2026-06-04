"""Perceptual auxiliary losses for Fizgig Klein 9B LoRA training.

Phase 1 — Depth consistency (DA2):        depth_consistency.py
Phase 2 — Identity / landmark losses:     face_id.py
Phase 3 — Subject masks (YOLO+SAM2):      subject_mask.py
Phase 4 — Body proportion (ViTPose):       body_id.py

Config dataclasses:  config.py
Dataset adapter:     adapter.py
"""
