"""Adapter bridging Fizgig's ItemInfo to the ai-toolkit FileItemDTO interface.

The perceptual caching functions (cache_depth_gt_embeddings, cache_face_embeddings,
cache_subject_masks, cache_body_proportion_embeddings) were written for ai-toolkit's
FileItemDTO objects. PerceptualAdapter duck-types that interface using Fizgig's
ItemInfo as the backing store.

Usage:
    adapters = build_perceptual_adapters(train_dataset_group)
    cache_depth_gt_embeddings(adapters, depth_config, device=device, vae_roundtrip_fn=fn)
    for a in adapters:
        a.sync()
    # After sync(), ItemInfo objects have is_depth_cached=True and _depth_cache_path set.
    # BucketBatchManager.__getitem__ reads these to populate batch['depth_gt'].
"""

import os
from typing import List, Optional, Tuple

from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS, ItemInfo


def _compute_resize_crop_params(
    orig_w: int, orig_h: int, bucket_w: int, bucket_h: int
) -> Tuple[int, int, int, int]:
    """Return (scaled_w, scaled_h, crop_x, crop_y) matching resize_image_to_bucket()."""
    scale = max(bucket_w / orig_w, bucket_h / orig_h)
    new_w = int(orig_w * scale + 0.5)
    new_h = int(orig_h * scale + 0.5)
    crop_x = (new_w - bucket_w) // 2
    crop_y = (new_h - bucket_h) // 2
    return new_w, new_h, crop_x, crop_y


class PerceptualAdapter:
    """Duck-types ai-toolkit's FileItemDTO for perceptual caching functions.

    After the caching pass, call sync() to propagate cache metadata back to
    the underlying ItemInfo so BucketBatchManager can find the sidecar files.
    """

    def __init__(self, item_info: ItemInfo, image_path: str, bucket_reso: Tuple[int, int]) -> None:
        self._item_info = item_info
        self.path = image_path

        bucket_w, bucket_h = bucket_reso
        orig_w, orig_h = item_info.original_size

        if orig_w > 0 and orig_h > 0 and bucket_w > 0 and bucket_h > 0:
            new_w, new_h, crop_x, crop_y = _compute_resize_crop_params(
                orig_w, orig_h, bucket_w, bucket_h
            )
            self.scale_to_width = new_w
            self.scale_to_height = new_h
            self.crop_x = crop_x
            self.crop_y = crop_y
            self.crop_width = bucket_w
            self.crop_height = bucket_h
        else:
            self.scale_to_width = None
            self.scale_to_height = None
            self.crop_x = None
            self.crop_y = None
            self.crop_width = None
            self.crop_height = None

        self.flip_x = False
        self.flip_y = False

        # --- Depth cache metadata (populated by cache_depth_gt_embeddings) ---
        self.depth_gt = None
        self._depth_cache_path: Optional[str] = None
        self._depth_cache_key: Optional[str] = None
        self.is_depth_cached: bool = False

        # --- Face / identity cache (populated by cache_face_embeddings) ---
        self.face_embedding = None
        self.face_bbox = None
        self.identity_embedding = None
        self.landmark_embedding = None

        # --- Subject mask cache (populated by cache_subject_masks) ---
        self.subject_mask = None
        self.body_mask = None
        self.clothing_mask = None
        self._mask_cache_path: Optional[str] = None
        self.is_mask_cached: bool = False

        # --- Body proportion cache (populated by cache_body_proportion_embeddings) ---
        self.body_proportion_embedding = None
        self.is_body_proportion_cached: bool = False

    def sync(self) -> None:
        """Copy all cache metadata back to the underlying ItemInfo."""
        ii = self._item_info

        # Depth — lazy loading: BucketBatchManager reads from sidecar on demand
        ii._depth_cache_path = self._depth_cache_path
        ii._depth_cache_key = self._depth_cache_key
        ii.is_depth_cached = self.is_depth_cached

        # Face embeddings — small tensors, keep in-memory on ItemInfo
        if self.identity_embedding is not None:
            ii.identity_embedding = self.identity_embedding
        if self.face_bbox is not None:
            ii.face_bbox = self.face_bbox
        if self.landmark_embedding is not None:
            ii.landmark_embedding = self.landmark_embedding

        # Subject mask — cache_subject_masks() sets these directly on the adapter
        # (it's passed the adapter list as its "file_items"), never the
        # _mask_cache_path/is_mask_cached pair below — those are legacy/unused by
        # the current cache_subject_masks(), kept only so stale readers don't KeyError.
        if self.subject_mask is not None:
            ii.subject_mask = self.subject_mask
        if self.body_mask is not None:
            ii.body_mask = self.body_mask
        if self.clothing_mask is not None:
            ii.clothing_mask = self.clothing_mask
        ii._mask_cache_path = self._mask_cache_path
        ii.is_mask_cached = self.is_mask_cached

        # Body proportion — small tensor, keep in-memory
        if self.body_proportion_embedding is not None:
            ii.body_proportion_embedding = self.body_proportion_embedding
        ii.is_body_proportion_cached = self.is_body_proportion_cached


def _resolve_image_path(image_dir: str, item_key: str) -> Optional[str]:
    """Find the image file for a given item_key (filename stem) in image_dir."""
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(image_dir, item_key + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def build_perceptual_adapters(train_dataset_group) -> List[PerceptualAdapter]:
    """Build one PerceptualAdapter per unique (item_key, bucket_reso) pair.

    Deduplicates across num_repeats copies and across DatasetGroup members.
    """
    seen: set = set()
    adapters: List[PerceptualAdapter] = []

    for ds in train_dataset_group.datasets:
        if ds.batch_manager is None:
            continue
        for bucket_reso, items in ds.batch_manager.buckets.items():
            for item_info in items:
                key = (item_info.item_key, bucket_reso)
                if key in seen:
                    continue
                seen.add(key)

                image_path = _resolve_image_path(ds.image_directory, item_info.item_key)
                if image_path is None:
                    continue

                adapters.append(PerceptualAdapter(item_info, image_path, bucket_reso))

    return adapters
