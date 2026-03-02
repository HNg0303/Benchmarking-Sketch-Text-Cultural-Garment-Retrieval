# 👗 AoDai-STC: Synthesizing Cultural Garment Triplets for Composed Image Retrieval

**Official repository for AoDai-STC**, a comprehensive pipeline for synthesizing high-quality triplet datasets (sketch-text-image) specifically for cultural garments. We address the data scarcity bottleneck in Composed Image Retrieval (CIR) by leveraging state-of-the-art generative models to produce **20,000 curated triplets** of the Vietnamese áo dài.

## 📋 Overview

This research project presents a novel pipeline combining **SANA-ControlNet** for high-fidelity sketch-conditioned image synthesis with **Qwen2.5 3B Instruct** for semantic caption generation. We address the critical challenge of obtaining diverse, high-quality training data for specialized domains like cultural garments. Our approach automates the entire triplet synthesis workflow, eliminating manual annotation and producing a large-scale, multi-modal dataset suitable for training and evaluation of composed image retrieval systems.

## 🎯 Motivation

Composed image retrieval (CIR) is the task of finding images in a database based on a query combining a reference image with text modifications. However, obtaining diverse, high-quality training data remains a significant bottleneck, particularly for specialized domains like cultural garments where data is scarce.

This project addresses this challenge by:
- **Automating triplet generation** from human-drawn sketch inputs, eliminating manual annotation requirements
- **Leveraging SANA-ControlNet** to synthesize photorealistic, culturally accurate garment images conditioned on spatial sketches
- **Utilizing Qwen2.5 3B Instruct** to generate consistent, attribute-rich captions starting with "A photo of..."
- **Modeling real-world uncertainty** via a Multi-Target Query Design ($1 \rightarrow 3$ mapping) that addresses false-negative supervision in composed retrieval
- **Producing large-scale datasets** (20,000 triplets) that enable training of robust CIR models for cultural garment applications

## 🔧 Technical Approach

### The Synthesis Pipeline

Our pipeline follows a structured flow from abstract sketch input to aligned multi-modal triplets:

```
Sketch Input  +  Attribute Sampling  +  Structured Prompts
    ↓                      ↓
[SANA-ControlNet: Spatial Control]  [Attribute Generation]
    ↓
Synthesized High-Fidelity Garment Image
    ↓
[Qwen2.5 3B Instruct: Caption Distillation]
    ↓
Neutral, Factual Text Description (Starting with "A photo of...")
    ↓
Triplet Formation: (Sketch, Caption, Multi-Target Images)
    ↓
Curated Triplet Dataset: (S, C, I₁, I₂, I₃)
```

### Key Components

1. **Attribute Sampling**
   - 11 curated categories from fashion archives: Fabric, Silhouette, Neckline, Sleeve Style, Color, Pattern, Embroidery, Fit, Length, Collar Type, and Ornamental Details
   - Structured sampling ensures diverse, culturally authentic garment representations
   - Maintains semantic consistency across triplets

2. **Image Synthesis (SANA-ControlNet)**
   - Generates high-quality garment images conditioned on sketch inputs and structured prompts
   - Spatial control via ControlNet ensures sketch-image semantic alignment
   - Produces diverse variations suitable for CIR training with photorealistic details
   - Maintains cultural authenticity while adding photorealistic rendering

3. **Caption Generation (Qwen2.5 3B Instruct)**
   - Distills the attribute set into neutral, factual single-sentence captions
   - All captions follow the format: "A photo of [garment description]..."
   - Captures essential attributes (style, color, fit, cultural elements) comprehensively
   - Produces descriptions that complement sketch-image pairs for robust triplet learning

4. **Triplet Formation & Multi-Target Query Design**
   - Aggregates Sketch (S), Caption (C), and Multiple Target Images (I₁, I₂, I₃) into final dataset entries
   - One-to-Three mapping ($1 \rightarrow 3$) explicitly addresses false-negative supervision in CIR
   - Provides multiple valid targets for each query, reflecting real-world retrieval scenarios
   - Creates balanced datasets with diverse cultural garment categories

## 📊 Dataset

### Dataset Statistics
- **Total Triplets**: 20,000 curated sketch-text-image triplets
- **Human Sketches**: 650 unique human-drawn query sketches
- **Synthesized Images**: 21,000 high-resolution fashion renderings (3 per triplet)
- **Domain**: Vietnamese Cultural Garments (Áo Dài and related traditional clothing)
- **Format**: JSON-based annotations with split support (train.json, val.json, test.json)
- **Modalities**: Sketch, Image, Text with explicit one-to-three mapping

### Data Organization
- **Triplet Format**: Each entry contains (Sketch S, Caption C, Image₁ I₁, Image₂ I₂, Image₃ I₃)
- **Splits**: Training, validation, and test sets with balanced class distributions
- **Annotations**: JSON files in `aodai/captions/` directory with image-sketch mapping files
- **Image Quality**: High-resolution photorealistic renderings suitable for deep learning models

## 📁 Project Structure

```
BM_ICMR2026/
├── generation/              # Image and caption generation pipeline
│   ├── sana_inference.py   # Single image synthesis script
│   ├── sana_inference_multi.py  # Batch synthesis script
│   ├── Sana/               # SANA model repository
│   ├── prompts_ao_dai.json # Generation prompts
│   └── features.json       # Feature storage
│
├── notebook/               # Jupyter notebooks for analysis
│   └── Qwen2.5_captions.ipynb  # Caption generation notebook
│
├── aodai/                  # Dataset directory
│   ├── origin.json         # Original sketch-text pairs
│   ├── output_triplet.json # Generated triplets
│   ├── train.json, test.json, val.json  # Split annotations
│   ├── captions/           # Caption data
│   │   ├── cap.train.json, cap.test.json, cap.val.json
│   │   └── triplet.json    # Final triplet annotations
│   ├── images/             # Synthesized garment images
│   ├── sketches/           # Input sketches
│   └── images_split/       # Image-sketch mapping files
│
├── benchmark/              # Evaluation benchmarks
│   ├── Bi-Blip4CIR/        # BliP-based CIR model
│   ├── CLIP4Cir/           # CLIP-based CIR model
│   ├── pic2word/           # Picture-to-word baseline
│   └── SEARLE/             # SEARLE benchmark
│
├── generated/              # Final outputs
│   ├── output_triplet_final.json      # Final triplet dataset
│   └── outputs_ao_dai_caption_refined.json
│
├── metric.py               # Evaluation metrics
├── process_triplets.py     # Triplet processing utilities
├── triplet.py              # Triplet data structures
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- See `requirements.txt` for complete dependency list

### Installation

1. **Clone the repository and install dependencies**
```bash
cd BM_ICMR2026
pip install -r requirements.txt
```

2. **Setup SANA model** (if generating new images)
```bash
cd generation/Sana
# Follow SANA installation instructions in generation/Sana/README.md
```

3. **Prepare input data**
```bash
# Ensure sketches are in aodai/sketches/
# Ensure prompts are configured in generation/prompts_ao_dai.json
```

## 📝 Usage

### 1. Image Synthesis with SANA

**Single image synthesis:**
```bash
cd generation
python sana_inference.py --config configs/sana_config.yaml --sketch_path path/to/sketch.png
```

**Batch synthesis:**
```bash
cd generation
bash inference_multi.sh
```

### 2. Caption Generation with Qwen2.5

Generate captions for synthesized images using the provided notebook:
```
notebook/Qwen2.5_captions.ipynb
```

Or use the caption generation script directly (if available).

### 3. Triplet Formation

Process and form triplets from synthesized images and captions:
```bash
python process_triplets.py --input_dir aodai/ --output_file generated/output_triplet_final.json
```

### 4. Evaluation

Evaluate CIR models on the synthesized dataset:
```bash
python metric.py --triplet_file aodai/captions/triplet.json --model_name CLIP4Cir
```

## 🏆 Research Contributions

1. **Automated Triplet Synthesis**: First pipeline combining diffusion spatial control (SANA-ControlNet) with LLM semantic distillation (Qwen2.5) for cultural garment domain-specific data generation

2. **Domain-Specific Benchmark**: Establishes the first comprehensive benchmark for fine-grained composed image retrieval in the "Áo Dài" (Vietnamese cultural garment) domain with 20,000 high-quality triplets

3. **One-to-Many Mapping Design**: Explicitly addresses false-negative supervision in composed retrieval through a structured Multi-Target Query Design ($1 \rightarrow 3$ mapping) that reflects real-world uncertainty

4. **Multi-Modal Consistency**: Ensures semantic alignment across sketch, image, and text modalities through structured attribute sampling and LLM caption distillation

5. **Benchmark Evaluation**: Comprehensive evaluation against state-of-the-art CIR models (Bi-Blip4CIR, CLIP4Cir, Pic2Word, SEARLE) demonstrating dataset utility

## 📚 Benchmarks

The project includes implementations and evaluations of:

- **Bi-Blip4CIR**: Bidirectional BLIP-based composed image retrieval
- **CLIP4Cir**: CLIP-based composed image retrieval with fine-tuning
- **Pic2Word**: Vision-language model for composed retrieval
- **SEARLE**: Scalable end-to-end architecture for image retrieval

Each benchmark includes pre-trained models and evaluation scripts in the `benchmark/` directory.

## � Citation

If you use this dataset or pipeline in your research, please cite:

```bibtex
@misc{AoDaiSTC2026,
  title={Synthesizing Cultural Garment Triplets for Composed Image Retrieval},
  author={[Your Name/Lab]},
  year={2026},
  note={Submitted to ICMR 2026}
}
```

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out to the project maintainers.

## 📖 References

- SANA: Efficient High-Resolution Image Synthesis
- Qwen2.5: Advanced Large Language Model
- CLIP4Cir: Composed Image Retrieval with Fine-tuning
- Composed Image Retrieval: A Review and Best Practices

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Status**:  Research Project  
**Last Updated**: March 2026  
**Version**: 1.0
