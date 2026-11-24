# One-Shot Image Classification Using Confidence-Adaptive Foundation Model Fusion

## Overview

This project investigates one-shot image classification using
frozen CLIP and SigLIP 2 foundation models.

The objective is to maximize classification accuracy using
strictly one original training image per class.

The proposed method combines semantic text representations
and augmented one-shot visual prototypes through a
confidence-adaptive four-expert fusion mechanism.

## Key Results

| Metric | Result |
|---|---:|
| Number of classes | 16 |
| Original training images | 16 |
| Training images per class | 1 |
| Test images | 400 |
| Experiments | 16 |
| Best observed accuracy | 92.50% |
| Correct predictions | 370 |
| Incorrect predictions | 30 |
| Independent prediction agreement | 100% |

## Methodology

### Foundation Models

- CLIP ViT-L/14: 768-dimensional embeddings.
- SigLIP 2 SO400M Patch14: 1152-dimensional embeddings.
- Both foundation models remain frozen.
- No foundation model fine-tuning is performed.

### One-Shot Prototype Learning

One original training image is randomly selected per class
using seed 42.

Nine views are generated from each selected image using
horizontal flipping, center cropping, brightness, contrast,
and gamma transformations.

These augmented views are used to construct visual prototypes.
No additional original training images are used for learning.

### Confidence-Adaptive Fusion

Four classification experts are combined:

1. CLIP semantic expert.
2. CLIP visual prototype expert.
3. SigLIP semantic expert.
4. SigLIP visual prototype expert.

Expert scores are standardized independently for each image.

Confidence is calculated as the difference between the two
highest standardized class scores.

The adaptive weights are calculated using:

    weight_i = softmax(log(prior_i) + beta * confidence_i)

Initial expert priors:

    [0.35, 0.15, 0.35, 0.15]

Confidence beta:

    1.0

The final classifier uses augmented visual prototypes for
both foundation models.

## Experimental Comparison

| Experiment | Accuracy |
|---|---:|
| CLIP Zero-Shot | 88.25% |
| CLIP One-Shot | 75.50% |
| CLIP Fusion | 89.75% |
| CLIP Augmented One-Shot | 75.50% |
| CLIP Augmented Fusion | 90.00% |
| SigLIP Zero-Shot | 88.00% |
| SigLIP One-Shot | 78.25% |
| SigLIP Fusion | 91.50% |
| SigLIP Augmented One-Shot | 77.75% |
| SigLIP Augmented Fusion | 91.50% |
| Ensemble Zero-Shot | 88.75% |
| Ensemble One-Shot | 78.75% |
| Ensemble Fusion | 92.25% |
| Confidence-Adaptive Fusion (E14) | **92.50%** |
| TTA Fixed Fusion | 92.25% |
| TTA Adaptive Fusion | **92.50%** |

E14 is selected as the final classifier because it achieves
the highest observed accuracy without additional test-time
augmentation.

## Installation

Use a Python environment with PyTorch and the required
dependencies.

```bash
pip install -r requirements.txt
```

A CUDA-compatible GPU is recommended.

Install a compatible PyTorch and torchvision build for
your operating system and CUDA environment if necessary.

The pretrained foundation model weights are downloaded
automatically during inference.

## Single-Image Inference

```bash
python inference.py \
    --checkpoint model/e14_final_classifier.pt \
    --image path/to/image.jpg \
    --device cuda
```

The script returns the predicted class, fused classification
score, and adaptive expert weights.

## Dataset Evaluation

Obtain the original assignment test dataset separately
and organize it into class-specific directories:

```text
dataset/test/
    Bedroom/
    Coast/
    Flower/
    Forest/
    Highway/
    Industrial/
    InsideCity/
    Kitchen/
    LivingRoom/
    Mountain/
    Office/
    OpenCountry/
    Store/
    Street/
    Suburb/
    TallBuilding/
```

Run:

```bash
python inference.py \
    --checkpoint model/e14_final_classifier.pt \
    --test-dir dataset/test \
    --device cuda \
    --clip-batch 16 \
    --siglip-batch 8 \
    --output-csv results/reproduced_predictions.csv
```

Expected results in the tested environment:

```text
Total images: 400
Correct predictions: 370
Incorrect predictions: 30
Accuracy: 92.50%
```

## Reproducibility

The standalone inference implementation was executed in
an independent Python process using freshly extracted
image embeddings.

It reproduced all 400 original E14 predictions exactly.

The fitted classifier checkpoint contains semantic
embeddings, visual prototypes, class mappings, foundation
model identifiers, and adaptive fusion configuration.

The original foundation model weights are not included
because they remain frozen and are loaded from their
pretrained sources.

Dataset provenance and selected training sample information
are provided in the metadata directory.

## Limitations

- Only one randomly selected support image per class is used.
- Visually similar scene categories remain challenging.
- Foundation model inference requires substantial compute.
- The best method was selected by comparing performance
  on the supplied test set; the reported accuracy is the
  highest observed test accuracy, not an independent
  estimate on a separate unseen evaluation dataset.
- Numerical results may vary across hardware and software
  environments.

## Conclusion

Confidence-adaptive fusion combines complementary semantic
and visual information from two frozen foundation models.

The proposed classifier achieves 92.50% observed accuracy
using only 16 original training images, with exact
image-level prediction reproducibility in the tested
environment.

## References

- Radford et al. Learning Transferable Visual Models
  From Natural Language Supervision. ICML, 2021.
- Tschannen et al. SigLIP 2: Multilingual Vision-Language
  Encoders with Improved Semantic Understanding,
  Localization, and Dense Features. 2025.

## Academic Context

Computer Vision — Assignment 3

University of North Carolina at Charlotte
Fall 2026
