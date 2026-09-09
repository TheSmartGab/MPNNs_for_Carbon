"""Utility to load and transfer state dictionaries from .pt checkpoints onto MACE model templates.

This module provides a utility script for loading PyTorch checkpoint (.pt) files onto existing
MACE model template (.model) files, handling dimension mismatches that can occur after certain
training operations like multi-head fine-tuning or replay set training.

Functions:
    parse_args: Parse command-line arguments
    main: Main entry point to load checkpoints and save updated models
"""
import torch
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    """Parse command-line arguments for the checkpoint loading utility.

    Returns:
        Namespace: Parsed arguments with 'template', 'checkpoint', and 'device' fields.
    """
    parser = ArgumentParser(description="Utility to load .pt checkpoints onto template .model files.")

    parser.add_argument("--template", "-t", help="Path to the MACE model template (.model file).")
    parser.add_argument("--checkpoint", "-c", help="Path to the checkpoint file (.pt file).")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu", "mps"], default="cpu",
                        help="Device to load tensors onto (default: cpu)")

    args = parser.parse_args()

    return args


def main():
    """Main entry point for the checkpoint loading utility.

    This function reads a MACE model template and a PyTorch checkpoint, aligns their
    state dictionaries (handling dimension mismatches where the checkpoint has an extra
    leading dimension like [1, 128] but the template expects [128]), and saves the updated
    model as a new .model file.

    Returns:
        int: 0 on success.
    """

    args = parse_args()

    template_path = Path(args.template)
    checkpoint_path = Path(args.checkpoint)

    print("[INFO] Reading", template_path)
    template = torch.load(template_path, map_location=torch.device(args.device), weights_only=False)
    print("[INFO] Reading", checkpoint_path)
    checkpoints = torch.load(checkpoint_path, map_location=torch.device(args.device), weights_only=False)

    state_dict = checkpoints['model']
    template_state_dict = template.state_dict()
    updated_state_dict = {}

    for key, value in state_dict.items():
        if key in template_state_dict:
            target_shape = template_state_dict[key].shape
            # If the checkpoint has an extra dimension (e.g., [1, 128]) but the template wants [128]
            if value.shape != target_shape and value.shape == (1, *target_shape):
                print(f"[FIX] Squeezing key: {key} from {value.shape} to {target_shape}")
                value = value.squeeze(0)
        updated_state_dict[key] = value

    template.load_state_dict(updated_state_dict, strict=False)

    out_model = checkpoint_path.with_suffix(".model")

    print("[INFO] Saving to", out_model)
    torch.save(template, out_model)

    return 0


if __name__ == "__main__":
    main()