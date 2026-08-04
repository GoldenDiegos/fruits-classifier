"""Gradio demo for interactive fruit classification (Parcial 3 applicable use case).

Usage:
    python scripts/gradio_demo.py --model-path models/checkpoints/resnet18_model.pt --share
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment.inference import FruitPredictor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the fruit classifier Gradio demo")
    parser.add_argument("--model-path", default="models/checkpoints/resnet18_model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--share", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FruitPredictor(model_path=args.model_path, device=args.device)

    def classify(image_path):
        result = predictor.predict(image_path)
        return result["class"], result["probabilities"]

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="filepath", label="Fruit photo"),
        outputs=[
            gr.Label(label="Prediction"),
            gr.Label(label="Class probabilities", num_top_classes=5),
        ],
        title="Fruit Classifier",
        description="Upload a photo of an apple, banana, grape, mango, or strawberry.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
