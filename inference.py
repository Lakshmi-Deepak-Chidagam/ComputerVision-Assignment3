
import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import open_clip

from PIL import Image
from transformers import AutoModel, AutoProcessor
from torchvision.datasets import ImageFolder


class E14Classifier:

    def __init__(self, checkpoint_path, device=None):

        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self.classes = self.checkpoint["classes"]
        self.fusion = self.checkpoint["fusion"]

    def release_memory(self):

        gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def load_image(path):

        with Image.open(path) as image:
            return image.convert("RGB")

    @torch.inference_mode()
    def extract_clip_features(self, image_paths, batch_size=16):

        config = self.checkpoint["clip"]

        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                config["model_name"],
                pretrained=config["pretrained"],
                device=self.device,
                precision=(
                    "fp16"
                    if self.device.type == "cuda"
                    else "fp32"
                )
            )
        )

        model.eval()
        model.requires_grad_(False)

        input_dtype = model.visual.conv1.weight.dtype
        features = []

        try:

            for start in range(
                0, len(image_paths), batch_size
            ):

                paths = image_paths[start:start + batch_size]

                batch = torch.stack([
                    preprocess(self.load_image(path))
                    for path in paths
                ]).to(
                    device=self.device,
                    dtype=input_dtype
                )

                output = model.encode_image(batch)

                output = F.normalize(
                    output.float(), dim=-1
                )

                features.append(output.cpu())

                del batch, output

        finally:

            del model
            self.release_memory()

        return torch.cat(features, dim=0)

    @torch.inference_mode()
    def extract_siglip_features(self, image_paths, batch_size=8):

        config = self.checkpoint["siglip"]

        processor = AutoProcessor.from_pretrained(
            config["model_id"]
        )

        model = AutoModel.from_pretrained(
            config["model_id"],
            torch_dtype=self.dtype
        ).to(self.device)

        model.eval()
        model.requires_grad_(False)

        features = []

        try:

            for start in range(
                0, len(image_paths), batch_size
            ):

                paths = image_paths[start:start + batch_size]

                images = [
                    self.load_image(path)
                    for path in paths
                ]

                inputs = processor(
                    images=images,
                    return_tensors="pt"
                )

                pixel_values = inputs["pixel_values"].to(
                    device=self.device,
                    dtype=self.dtype
                )

                output = model.get_image_features(
                    pixel_values=pixel_values
                )

                if not isinstance(output, torch.Tensor):
                    output = output.pooler_output

                output = F.normalize(
                    output.float(), dim=-1
                )

                features.append(output.cpu())

                del inputs, pixel_values, output

        finally:

            del model, processor
            self.release_memory()

        return torch.cat(features, dim=0)

    def standardize(self, scores):

        epsilon = self.fusion["standardization_epsilon"]

        return (
            scores - scores.mean(dim=1, keepdim=True)
        ) / (
            scores.std(dim=1, keepdim=True) + epsilon
        )

    @torch.inference_mode()
    def fuse_features(self, clip_features, siglip_features):

        clip_features = clip_features.cpu().float()
        siglip_features = siglip_features.cpu().float()

        clip = self.checkpoint["clip"]
        siglip = self.checkpoint["siglip"]

        selection = self.fusion["prototype_selection"]

        expert_scores = torch.stack([

            self.standardize(
                clip_features @ clip["text_features"].T
            ),

            self.standardize(
                clip_features @ clip[selection["clip"]].T
            ),

            self.standardize(
                siglip_features @ siglip["text_features"].T
            ),

            self.standardize(
                siglip_features @ siglip[selection["siglip"]].T
            )

        ], dim=1)

        top_two = torch.topk(
            expert_scores, k=2, dim=2
        ).values

        confidence = (
            top_two[:, :, 0] - top_two[:, :, 1]
        )

        priors = self.fusion["prior_weights"]
        beta = self.fusion["confidence_beta"]

        weights = torch.softmax(
            torch.log(priors) + beta * confidence,
            dim=1
        )

        fused_scores = (
            expert_scores * weights.unsqueeze(-1)
        ).sum(dim=1)

        predictions = fused_scores.argmax(dim=1)

        return predictions, fused_scores, weights

    def predict(self, image_paths, clip_batch=16, siglip_batch=8):

        image_paths = list(image_paths)

        if not image_paths:
            raise ValueError("No input images provided.")

        print("Extracting CLIP features...")

        clip_features = self.extract_clip_features(
            image_paths, clip_batch
        )

        print("Extracting SigLIP features...")

        siglip_features = self.extract_siglip_features(
            image_paths, siglip_batch
        )

        return self.fuse_features(
            clip_features,
            siglip_features
        )


def main():

    parser = argparse.ArgumentParser(
        description="E14 One-Shot Image Classifier"
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--clip-batch", type=int, default=16)
    parser.add_argument("--siglip-batch", type=int, default=8)

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument("--image")
    mode.add_argument("--test-dir")

    args = parser.parse_args()

    classifier = E14Classifier(
        args.checkpoint,
        args.device
    )

    print("Device:", classifier.device)

    if args.image:

        predictions, scores, weights = classifier.predict(
            [args.image],
            args.clip_batch,
            args.siglip_batch
        )

        index = predictions[0].item()

        result = {
            "image": str(args.image),
            "predicted_class": classifier.classes[index],
            "class_index": index,
            "fused_score": scores[0, index].item(),
            "expert_weights": {
                name: float(weight)
                for name, weight in zip(
                    classifier.fusion["expert_order"],
                    weights[0]
                )
            }
        }

        print(json.dumps(result, indent=2))

    else:

        dataset = ImageFolder(args.test_dir)

        if dataset.classes != classifier.classes:
            raise ValueError(
                "Dataset classes do not match checkpoint classes."
            )

        paths = [path for path, _ in dataset.samples]

        labels = torch.tensor([
            label for _, label in dataset.samples
        ])

        predictions, _, _ = classifier.predict(
            paths,
            args.clip_batch,
            args.siglip_batch
        )


        if args.output_csv:

            output_path = Path(args.output_csv)
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            with output_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "image_path",
                    "true_index",
                    "true_class",
                    "predicted_index",
                    "predicted_class",
                    "correct"
                ])

                for path, label, prediction in zip(
                    paths, labels.tolist(), predictions.tolist()
                ):

                    writer.writerow([
                        Path(path).relative_to(args.test_dir).as_posix(),
                        label,
                        classifier.classes[label],
                        prediction,
                        classifier.classes[prediction],
                        int(label == prediction)
                    ])

            print("Prediction CSV saved:", output_path)

        correct = (predictions == labels).sum().item()
        accuracy = correct / len(labels) * 100

        print("\n" + "=" * 50)
        print("STANDALONE EVALUATION RESULTS")
        print("=" * 50)
        print("Total images:", len(labels))
        print("Correct predictions:", correct)
        print("Incorrect predictions:", len(labels) - correct)
        print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
