#!/usr/bin/env python
"""
Fizgig Preflight
================
Validates a fat training config (produced by Fizgig's Save Config button) and
ensures every external dependency is present before a headless training run.

What it checks:
  - Config file is valid TOML and contains required sections
  - Dataset images and caption files are present
  - Fizgig training module is importable
  - All model files exist; downloads any that are missing via huggingface_hub
  - Patches the config in-place if a downloaded model landed at a different path

HF authentication:
  Checks for a cached token before any downloads.  Gated models (FLUX.2 family)
  require auth.  If no token is found the user is offered a standard
  'huggingface-cli login' prompt — credentials are never handled by this script.

Usage:
  python preflight.py path/to/train_config.toml
"""

import argparse
import os
import sys
import glob
import importlib.util

# ---------------------------------------------------------------------------
# Model registry — maps filename stems to HF repo + filename + gated flag
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "flux-2-klein-base-9b-fp8": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9b-fp8",
        "filename": "flux-2-klein-base-9b-fp8.safetensors",
        "gated": True,
    },
    "flux-2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "filename": "flux-2-klein-base-9b.safetensors",
        "gated": True,
    },
    "flux-2-klein-9b-fp8": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9b-fp8",
        "filename": "flux-2-klein-9b-fp8.safetensors",
        "gated": True,
    },
    "ae": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "ae.safetensors",
        "gated": True,
    },
    "qwen_3_8b": {
        "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "filename": "split_files/text_encoders/qwen_3_8b.safetensors",
        "gated": False,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(text):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print('─' * 60)

def _ok(msg):
    print(f"  ✓  {msg}")

def _warn(msg):
    print(f"  ⚠  {msg}")

def _fail(msg):
    print(f"  ✗  {msg}")

def _resolve(path, config_dir):
    """Resolve a path from the config — absolute stays absolute, relative resolves from config dir."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(config_dir, path))

def _stem(path):
    """Return the filename without extension, lower-cased."""
    return os.path.splitext(os.path.basename(path))[0].lower()

def _registry_entry(path):
    """Look up a model path in the registry by filename stem."""
    return MODEL_REGISTRY.get(_stem(path))

# ---------------------------------------------------------------------------
# Auth check
# ---------------------------------------------------------------------------

def check_hf_auth(needs_gated):
    """
    Check for a cached HF token.  If gated models are needed and no token is
    found, offer to run 'huggingface-cli login'.  Returns True if it's safe to
    proceed with downloads.
    """
    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        _fail("huggingface_hub is not installed — cannot download models.")
        sys.exit(1)

    if token:
        _ok("HuggingFace — authenticated")
        return True

    if needs_gated:
        _warn("No HuggingFace token found.  The following models are gated and require authentication.")
        answer = input("\n  Run 'huggingface-cli login' now? (y/n): ").strip().lower()
        if answer == "y":
            import subprocess
            subprocess.run(["huggingface-cli", "login"], check=False)
            # Re-check after login attempt
            token = get_token()
            if token:
                _ok("HuggingFace — authenticated")
                return True
            else:
                _fail("Still no token after login attempt.  Cannot download gated models.")
                sys.exit(1)
        else:
            _fail("Gated models cannot be downloaded without authentication.  Aborting.")
            sys.exit(1)
    else:
        _warn("No HuggingFace token — downloads will be anonymous (may be rate-limited).")
        answer = input("\n  Run 'huggingface-cli login' for faster downloads? (y/n): ").strip().lower()
        if answer == "y":
            import subprocess
            subprocess.run(["huggingface-cli", "login"], check=False)
        return True

# ---------------------------------------------------------------------------
# Validation steps
# ---------------------------------------------------------------------------

def load_config(config_path):
    _header("Loading config")
    try:
        import toml
    except ImportError:
        _fail("'toml' package not installed.")
        sys.exit(1)

    if not os.path.exists(config_path):
        _fail(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = toml.load(f)

    for required in ("models", "training", "output"):
        if required not in data:
            _fail(f"Config is missing required section [{required}]")
            sys.exit(1)

    _ok(f"Config loaded: {config_path}")
    return data


def check_dataset(data, config_dir):
    _header("Dataset")
    errors = 0

    image_dir = _resolve(data.get("datasets", [{}])[0].get("image_directory", ""), config_dir)
    if not image_dir:
        _fail("No image_directory found in [[datasets]]")
        return 1

    if not os.path.isdir(image_dir):
        _fail(f"Image directory not found: {image_dir}")
        return 1

    images = glob.glob(os.path.join(image_dir, "*.jpg")) + \
             glob.glob(os.path.join(image_dir, "*.jpeg")) + \
             glob.glob(os.path.join(image_dir, "*.png")) + \
             glob.glob(os.path.join(image_dir, "*.webp"))

    if not images:
        _fail(f"No images found in: {image_dir}")
        errors += 1
    else:
        _ok(f"{len(images)} image(s) found in {image_dir}")

    caption_ext = data.get("general", {}).get("caption_extension", ".txt")
    missing_captions = [f for f in images if not os.path.exists(os.path.splitext(f)[0] + caption_ext)]
    if missing_captions:
        _warn(f"{len(missing_captions)} image(s) have no {caption_ext} caption file")
        for f in missing_captions[:5]:
            print(f"       {os.path.basename(f)}")
        if len(missing_captions) > 5:
            print(f"       ... and {len(missing_captions) - 5} more")
        errors += 1
    else:
        _ok(f"All images have {caption_ext} captions")

    return errors


def check_fizgig():
    _header("Fizgig training module")
    # fizgig lives in src/ and is path-inserted at runtime rather than installed as a package
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.find_spec("fizgig.training.trainer")
    if spec is None:
        _fail("fizgig.training.trainer is not importable — is the venv active and src/ present?")
        return 1
    _ok("fizgig.training.trainer is importable")
    return 0


def check_and_download_models(data, config_dir):
    _header("Models")

    models_section = data.get("models", {})
    sampling_section = data.get("sampling", {})

    model_keys = [
        (models_section, "dit"),
        (models_section, "vae"),
        (models_section, "text_encoder"),
        (sampling_section, "sample_dit"),
    ]

    # Collect which entries need downloading and whether any are gated
    to_download = []
    needs_gated = False
    for section, key in model_keys:
        raw = section.get(key, "")
        if not raw:
            continue
        resolved = _resolve(raw, config_dir)
        if os.path.exists(resolved):
            _ok(f"{key}: {resolved}")
        else:
            entry = _registry_entry(resolved)
            if entry:
                to_download.append((section, key, raw, resolved, entry))
                if entry["gated"]:
                    needs_gated = True
                _warn(f"{key}: not found — will download  ({os.path.basename(resolved)})")
            else:
                _fail(f"{key}: not found and not in model registry: {resolved}")

    if not to_download:
        return data, 0

    # Auth check before any downloading
    check_hf_auth(needs_gated)

    from huggingface_hub import hf_hub_download

    patches = {}
    errors = 0
    for section, key, raw, resolved, entry in to_download:
        dest_dir = os.path.dirname(resolved) if os.path.isabs(resolved) else config_dir
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n  Downloading {entry['filename']} from {entry['repo_id']} ...")
        try:
            downloaded = hf_hub_download(
                repo_id=entry["repo_id"],
                filename=entry["filename"],
                local_dir=dest_dir,
            )
            actual = os.path.normpath(downloaded)
            _ok(f"Downloaded to: {actual}")
            if actual != os.path.normpath(resolved):
                patches[key] = actual
                _warn(f"Path differs from config — will patch config")
        except Exception as e:
            _fail(f"Download failed for {key}: {e}")
            errors += 1

    # Patch the config dict with corrected paths (caller will write back to disk)
    for key, actual_path in patches.items():
        if key in data.get("models", {}):
            data["models"][key] = actual_path
        elif key in data.get("sampling", {}):
            data["sampling"][key] = actual_path

    return data, errors


def write_patched_config(data, config_path):
    import toml
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(data, f)
    _ok(f"Config patched and saved: {config_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fizgig preflight — validate config before headless training")
    parser.add_argument("config", help="Path to fat training config TOML")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)

    print(f"\nFizgig Preflight")
    print(f"Config: {config_path}")

    data = load_config(config_path)

    errors = 0
    errors += check_dataset(data, config_dir)
    errors += check_fizgig()
    data, model_errors = check_and_download_models(data, config_dir)
    errors += model_errors

    if model_errors == 0 and data:
        # Write back only if patches were made (toml.dump is idempotent but let's be tidy)
        write_patched_config(data, config_path)

    _header("Summary")
    if errors == 0:
        print(f"\n  All checks passed.  Ready to train.\n")
        print(f"  Run:")
        print(f"    python -m fizgig.training.trainer --config_file \"{config_path}\"\n")
    else:
        print(f"\n  {errors} issue(s) found.  Resolve the above before training.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
