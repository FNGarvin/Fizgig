"""Configuration dataclasses for perceptual auxiliary losses.

Ported from ai-toolkit-perceptual/toolkit/config_modules.py.
Only the fields needed by the Fizgig integration are included.
"""
from typing import Optional, Union


class DepthConsistencyConfig:
    """Depth-consistency auxiliary loss via a frozen Depth-Anything-V2 perceptor.

    Enable by setting loss_weight > 0. The v3 cache stores GT depth extracted from
    VAE-roundtrip pixels so the loss floor can reach zero during training.
    """

    def __init__(self, **kwargs):
        self.loss_weight: float = kwargs.get('loss_weight', 0.1)
        self.loss_min_t: float = kwargs.get('loss_min_t', 0.0)
        self.loss_max_t: float = kwargs.get('loss_max_t', 1.0)
        self.model_id: str = kwargs.get('model_id', 'depth-anything/Depth-Anything-V2-Small-hf')
        self.input_size: int = kwargs.get('input_size', 518)
        self.pixel_blur_sigma: float = kwargs.get('pixel_blur_sigma', 0.0)
        self.ssi_weight: float = kwargs.get('ssi_weight', 1.0)
        self.grad_weight: float = kwargs.get('grad_weight', 0.5)
        self.grad_scales: int = kwargs.get('grad_scales', 4)
        self.mask_source: str = kwargs.get('mask_source', 'none')  # 'none'|'subject'|'body'
        self.grad_checkpoint: bool = kwargs.get('grad_checkpoint', True)
        self.preview_every: int = kwargs.get('preview_every', 0)
        self.preview_only: bool = kwargs.get('preview_only', False)


class FaceIDConfig:
    """Face identity and landmark auxiliary losses.

    identity_loss_weight > 0 enables ArcFace cosine identity loss.
    landmark_loss_weight > 0 enables MediaPipe FaceMesh landmark L1 loss.
    """

    def __init__(self, **kwargs):
        self.face_model: str = kwargs.get('face_model', 'buffalo_l')
        # --- ArcFace identity loss ---
        self.identity_loss_weight: float = kwargs.get('identity_loss_weight', 0.0)
        self.identity_loss_min_t: float = kwargs.get('identity_loss_min_t', 0.0)
        self.identity_loss_max_t: float = kwargs.get('identity_loss_max_t', 1.0)
        self.identity_loss_min_cos: float = kwargs.get('identity_loss_min_cos', 0.2)
        self.identity_loss_use_average: bool = kwargs.get('identity_loss_use_average', False)
        self.identity_loss_average_blend: float = kwargs.get('identity_loss_average_blend', 0.0)
        self.identity_loss_use_random: bool = kwargs.get('identity_loss_use_random', False)
        self.identity_loss_num_refs: int = kwargs.get('identity_loss_num_refs', 0)
        self.identity_metrics: bool = kwargs.get('identity_metrics', False)
        self.identity_loss_frames_per_chunk: int = kwargs.get('identity_loss_frames_per_chunk', 4)
        self.identity_loss_preview_every: int = kwargs.get('identity_loss_preview_every', 0)
        self.identity_loss_decoded_det_threshold: float = kwargs.get('identity_loss_decoded_det_threshold', 0.5)
        # --- MediaPipe landmark loss ---
        self.landmark_loss_weight: float = kwargs.get('landmark_loss_weight', 0.0)
        # --- Vision encoder (disabled by default in Fizgig) ---
        self.vision_enabled: bool = kwargs.get('vision_enabled', False)
        self.vision_model: str = kwargs.get('vision_model', 'openai/clip-vit-large-patch14')
        self.vision_num_tokens: int = kwargs.get('vision_num_tokens', 4)
        self.vision_crop_padding: float = kwargs.get('vision_crop_padding', 0.3)
        # --- Body proportion via face detector (ViTPose) ---
        self.body_proportion_loss_weight: float = kwargs.get('body_proportion_loss_weight', 0.0)
        self.body_proportion_loss_min_t: float = kwargs.get('body_proportion_loss_min_t', 0.0)
        self.body_proportion_loss_max_t: float = kwargs.get('body_proportion_loss_max_t', 1.0)
        self.body_proportion_include_head: bool = kwargs.get('body_proportion_include_head', False)
        self.body_proportion_frames_per_chunk: int = kwargs.get('body_proportion_frames_per_chunk', 2)
        self.body_proportion_preview_every: int = kwargs.get('body_proportion_preview_every', 0)
        # --- Body shape (HMR2/SMPL) — not used in Fizgig ---
        self.body_shape_loss_weight: float = kwargs.get('body_shape_loss_weight', 0.0)
        self.body_shape_loss_min_t: float = kwargs.get('body_shape_loss_min_t', 0.4)
        self.body_shape_loss_max_t: float = kwargs.get('body_shape_loss_max_t', 0.8)
        self.body_shape_loss_min_cos: float = kwargs.get('body_shape_loss_min_cos', 0.2)
        # --- Normal map loss — not used ---
        self.normal_loss_weight: float = kwargs.get('normal_loss_weight', 0.0)
        self.normal_loss_min_t: float = kwargs.get('normal_loss_min_t', 0.4)
        self.normal_loss_max_t: float = kwargs.get('normal_loss_max_t', 0.8)
        # --- Face suppression ---
        self.face_suppression_weight: Optional[float] = kwargs.get('face_suppression_weight', None)
        self.face_suppression_expand: float = kwargs.get('face_suppression_expand', 2.0)
        self.face_suppression_soft: bool = kwargs.get('face_suppression_soft', False)
        # --- VAE anchor loss — not used ---
        self.vae_anchor_loss_weight: float = kwargs.get('vae_anchor_loss_weight', 0.0)
        self.vae_anchor_loss_min_t: float = kwargs.get('vae_anchor_loss_min_t', 0.0)
        self.vae_anchor_loss_max_t: float = kwargs.get('vae_anchor_loss_max_t', 0.5)
        self.vae_anchor_model_path: str = kwargs.get('vae_anchor_model_path', '')


class SubjectMaskConfig:
    """Auto-masking via YOLO + SAM 2 + SegFormer-clothes."""

    def __init__(self, **kwargs):
        self.enabled: bool = kwargs.get('enabled', True)
        self.yolo_model: str = kwargs.get('yolo_model', 'yolo11x.pt')
        self.sam2_model: str = kwargs.get('sam2_model', 'facebook/sam2-hiera-large')
        self.segformer_model: str = kwargs.get('segformer_model', 'mattmdjaga/segformer_b2_clothes')
        self.mask_blur_sigma: float = kwargs.get('mask_blur_sigma', 3.0)
        self.person_confidence: float = kwargs.get('person_confidence', 0.5)
        self.expand_ratio: float = kwargs.get('expand_ratio', 0.1)
        self.body_mask_clothes: bool = kwargs.get('body_mask_clothes', True)


class BodyIDConfig:
    """Body shape conditioning config (HMR2/SMPL). Not used in Fizgig."""

    def __init__(self, **kwargs):
        self.enabled: bool = kwargs.get('enabled', False)
        self.num_tokens: int = kwargs.get('num_tokens', 4)
        self.dropout_prob: float = kwargs.get('dropout_prob', 0.1)
        self.detection_threshold: float = kwargs.get('detection_threshold', 0.5)
