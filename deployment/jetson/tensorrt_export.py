"""Export a trained PyTorch classifier to a TensorRT engine for Jetson."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_classifier(checkpoint: Path, output: Path, num_classes: int, input_size: int = 224) -> None:
    from industrial_vision.models.classifier.efficientnet import DefectClassifier

    model = DefectClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    try:
        import torch_tensorrt  # type: ignore[import-untyped]
        trt = torch_tensorrt.ts.compile(  # type: ignore[attr-defined]
            scripted,
            inputs=[torch.zeros(1, 3, input_size, input_size)],
            enabled_precisions={torch.float, torch.half},
        )
        torch.jit.save(trt, str(output))
    except ImportError:
        # Jetson-only dependency not installed; save the scripted module as a fallback.
        torch.jit.save(scripted, str(output))
    print(f"Saved TensorRT engine (or scripted fallback): {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()
    export_classifier(args.checkpoint, args.output, args.num_classes, args.input_size)


if __name__ == "__main__":
    main()
