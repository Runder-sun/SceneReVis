# SceneReVis

**SceneReVis: Iterative 3D Indoor Scene Generation with Vision-Language Reinforcement Learning**

A closed-loop framework for generating physically plausible and aesthetically coherent 3D indoor scenes through multi-turn iterative refinement. The system combines Vision-Language Model (VLM) reasoning, physics-based validation, and structured tool calls to produce high-quality 3D room layouts.

---

## 🏗️ Architecture Overview

SceneReVis operates through an iterative **Render → Evaluate → Revise** loop:

1. **Initial Scene Scaffolding**: Generate room boundaries and functional groups from text prompts
2. **Multi-modal Feedback Injection**: Combine physics feedback (collision/out-of-bounds detection via Trimesh) with VLM layout assessment
3. **Tool-based Scene Editing**: Structured `tool_calls` for `add_object`, `move_object`, `rotate_object`, `scale_object`, `replace_object`, `remove_object`, and `terminate`
4. **Asset Retrieval & Alignment**: Map abstract object descriptions to real 3D models (3D-FUTURE / Objaverse)
5. **Automated Rendering**: Blender-based dual-view rendering (top-down + diagonal perspective)

### Training Pipeline

- **SFT (Supervised Fine-Tuning)**: Train on CoT (Chain-of-Thought) conversation data with scene editing trajectories
- **RL (Reinforcement Learning)**: GRPO-based training with multi-turn scene editing interactions, using voxel-based physics rewards

---

## 📁 Project Structure

```
SceneReVis/
├── infer.py                      # Inference: iterative scene generation (single & batch)
│
├── eval/                         # Evaluation tools
│   ├── myeval.py                 # Mesh-based collision & OOB evaluation
│   ├── voxel_eval.py             # Voxel-based spatial evaluation
│   └── vlm_scene_eval.py         # VLM (GPT-4o Vision) multi-dimension evaluation
│
├── utils/                        # Core utilities
│   ├── sample.py                 # 3D-FUTURE asset retrieval (SigLIP-based)
│   ├── objaverse_retriever.py    # Objaverse asset retrieval (CLIP+SBERT)
│   ├── objaverse_glb_manager.py  # Objaverse GLB asset download & caching
│   ├── optimize_scene.py         # GPT-assisted scene physics optimization
│   ├── scene_editor.py           # Scene editing operations (add/remove/move/etc.)
│   ├── format_converter.py       # Scene format conversion (flat ↔ grouped)
│   ├── blender_renderer.py       # Blender rendering engine
│   ├── blender_wrapper.py        # Blender subprocess wrapper
│   ├── main_bpy.py               # Blender script entry point
│   ├── visualization_3d.py       # 3D visualization (bbox, arrows, grid)
│   ├── RL_utils.py               # RL training utilities
│   ├── path_config.py            # Unified path configuration manager
│   ├── image_merger.py           # Multi-view image composition
│   └── batch_render_all.py       # Batch rendering helper
│
├── script/                       # Training scripts
│   ├── RL/                       # Reinforcement learning
│   │   ├── scene_reward.py       # Reward function (voxel-based physics)
│   │   ├── scene_editing_interaction.py  # Multi-turn RL interaction handler
│   │   ├── run_grpo_B200.sh      # GRPO training launch script
│   │   └── config/               # RL configuration files
│   └── sft/                      # Supervised fine-tuning
│       └── sft_B200.sh           # SFT training launch script
│
├── verl/                         # VERL RL framework (modified fork)
│   └── verl/
│       ├── interactions/         # Multi-turn interaction interfaces
│       │   ├── base.py           # Base interaction class
│       │   └── scene_editing_interaction.py  # Scene editing interaction
│       ├── trainer/              # Training orchestration
│       └── ...
│
├── split_prompts/                # Test prompts (400 total across 4 room types)
│   ├── bedroom.txt               # 150 prompts
│   ├── living_room.txt           # 150 prompts
│   ├── dining_room.txt           # 50 prompts
│   └── study_room.txt            # 50 prompts
│
├── metadata/                     # Asset metadata
│   ├── model_info_3dfuture_assets.json
│   └── invalid_threed_front_rooms.txt
│
├── requirements_infer_batch.txt  # Inference dependencies
├── setup_env.sh                  # Environment variable setup
└── quick_install_blender.sh      # Blender 4.0.2 installation
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n scenerevis python=3.11 -y
conda activate scenerevis

# Install core dependencies
pip install ms-swift vllm accelerate deepspeed
pip install openai azure-identity
pip install trimesh scipy shapely pillow numpy
pip install compress_json compress_pickle open_clip_torch sentence-transformers
pip install swanlab msgspec python-fcl

# Or install from requirements file
pip install -r requirements_infer_batch.txt

# Install Blender 4.0.2 for rendering (no sudo required)
bash quick_install_blender.sh

# (Optional) Install VERL framework for RL training
cd verl && pip install -e . && cd ..
```

### 2. Download & Prepare Assets

#### 3D-FUTURE Models (Required)

1. Download the **3D-FUTURE** asset catalog from [Alibaba Tianchi](https://tianchi.aliyun.com/dataset/98063) (requires account + approval).
2. Download the **3D-FRONT** scene dataset from [Alibaba Tianchi](https://tianchi.aliyun.com/dataset/65347).
3. **Preprocessing**: Follow the preprocessing pipeline from [ReSpace](https://github.com/GradientSpaces/respace) to scale and prepare 3D-FUTURE assets:
   ```bash
   # Inside the ReSpace repo
   # 1. Set paths in .env file:
   #    PTH_3DFUTURE_ASSETS=/path/to/3D-FUTURE-model
   #    PTH_3DFRONT_SCENES=/path/to/3D-FRONT
   # 2. Scale assets to real-world sizes
   python ./src/preprocessing/3d-front/scale_assets.py
   # 3. Pre-compute asset embeddings for retrieval (or download the cached .pickle file)
   python ./src/preprocessing/3d-front/06_compute_embeds.py
   ```
   Alternatively, download the pre-computed embeddings cache (`model_info_3dfuture_assets_embeds.pickle`, ~174MB) from the ReSpace release.

#### Objaverse Assets & Annotations (Optional, for `--asset-source objaverse`)

Objaverse GLB models are downloaded on-demand during inference via `utils/objaverse_glb_manager.py`. However, you need the **Objaverse annotation files** for asset retrieval. Follow [Holodeck](https://github.com/allenai/Holodeck) to download them:

```bash
pip install objathor

# Download Objaverse annotations and features (saved to ~/.objathor-assets/ by default)
python -m objathor.dataset.download_annotations --version 2023_09_23
python -m objathor.dataset.download_features --version 2023_09_23

# (Optional) Download full Objaverse assets for offline use
python -m objathor.dataset.download_assets --version 2023_09_23
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
```

> **Note**: By default these save to `~/.objathor-assets/`. You can change the path via `--path` argument and set `OBJATHOR_ASSETS_BASE_DIR` environment variable accordingly.

#### Metadata Files
The `metadata/` directory contains JSON metadata for 3D-FUTURE assets. You also need the embeddings pickle file (`model_info_3dfuture_assets_embeds.pickle`) for asset retrieval — see the 3D-FUTURE section above for how to obtain it.

### 3. Configuration

```bash
# Set environment variables
source setup_env.sh

# Or set manually:
export PTH_3DFUTURE_ASSETS=/path/to/3D-FUTURE-model
export PTH_ASSETS_METADATA=./metadata/model_info_3dfuture_assets.json
export PTH_ASSETS_EMBED=./metadata/model_info_3dfuture_assets_embeds.pickle

# For Azure OpenAI (optional, for VLM feedback & initial room generation)
export AZURE_OPENAI_ENDPOINT=your_endpoint
export AZURE_OPENAI_SCOPE=your_scope
export AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Required for multi-GPU inference
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_RAY_SPMD_WORKER=0
export VLLM_USE_RAY_COMPILED_DAG=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
```

### 4. Inference

#### Mode 1: Model-based Scene Initialization (default)

The fine-tuned model generates the initial scene layout (room + objects), then iteratively refines it:

```bash
python infer.py \
    --prompt "Design a cozy bedroom with a queen bed and reading corner" \
    --model /path/to/checkpoint \
    --iterations 10 \
    --generate-room \
    --use-model-for-creation \
    --asset-source auto
```

#### Mode 2: GPT-based Object Initialization

Use GPT (Azure OpenAI) to generate the initial complete scene with furniture objects, then let the fine-tuned model refine it iteratively. This mode requires Azure OpenAI credentials to be configured:

```bash
python infer.py \
    --prompt "Design a cozy bedroom with a queen bed and reading corner" \
    --model /path/to/checkpoint \
    --iterations 10 \
    --generate-room \
    --use-gpt-with-objects \
    --asset-source auto
```

#### Mode 3: With Physics Optimization

Enable GPT-assisted physics optimization after each iteration to automatically resolve collisions and out-of-bounds issues. Can be combined with any initialization mode:

```bash
python infer.py \
    --prompt "Design a modern living room with a sectional sofa" \
    --model /path/to/checkpoint \
    --iterations 10 \
    --generate-room \
    --use-model-for-creation \
    --asset-source auto \
    --enable-physics-optimization \
    --physics-opt-steps 5 \
    --models-path /path/to/3D-FUTURE-model
```

You can also enable additional feedback injection:
- `--enable-physics-feedback`: Inject physics collision/OOB feedback into prompts
- `--enable-vlm-feedback`: Inject VLM layout assessment feedback into prompts (requires Azure OpenAI)

#### Mode 4: Batch Inference

Process multiple prompts from a file sequentially:

```bash
python infer.py \
    --batch-mode \
    --model /path/to/checkpoint \
    --prompts-file split_prompts/bedroom.txt \
    --output ./output/bedroom \
    --iterations 15 \
    --max-history-turns 8 \
    --asset-source objaverse \
    --generate-room \
    --use-model-for-creation \
    --skip-existing
```

### 5. Evaluation

```bash
# Collect final scenes from inference output
SCENES_DIR="./output/bedroom/final_scenes_collection"

# Mesh-based collision & OOB evaluation
python eval/myeval.py \
    --scenes_dir $SCENES_DIR \
    --models_path /path/to/3D-FUTURE-model \
    --output_dir ./output/bedroom/evaluation

# Voxel-based evaluation
python eval/voxel_eval.py \
    --scenes_dir $SCENES_DIR \
    --models_path /path/to/3D-FUTURE-model \
    --output_file ./output/bedroom/evaluation/voxel_results.json \
    --voxel_size 0.05

# VLM multi-dimension evaluation (requires Azure OpenAI)
python eval/vlm_scene_eval.py \
    --render-dir ./output/bedroom/rendered \
    --prompts-file split_prompts/bedroom.txt
```

### 6. Training

#### SFT (Supervised Fine-Tuning)

Training data: **SceneChain-12K** — 11,444 multi-turn scene editing conversation trajectories with rendered images.

```bash
# Run SFT training
bash script/sft/sft_B200.sh
```

#### RL (Reinforcement Learning with GRPO)

```bash
# Install VERL first
cd verl && pip install -e . && cd ..

# Run GRPO training
bash script/RL/run_grpo_B200.sh
```

---

## 📊 Evaluation Metrics

| Metric | Description | Tool |
|--------|-------------|------|
| Collision Rate | % of objects with physical overlaps | `myeval.py` / `voxel_eval.py` |
| Out-of-Bounds Rate | % of objects outside room boundaries | `myeval.py` / `voxel_eval.py` |
| VLM Rationality | Scene rationality score (0-10) | `vlm_scene_eval.py` |
| VLM Spatial Layout | Layout quality score (0-10) | `vlm_scene_eval.py` |
| VLM Accessibility | Accessibility score (0-10) | `vlm_scene_eval.py` |

---

## 🔧 Key Dependencies

- **[ms-swift](https://github.com/modelscope/ms-swift)**: Model inference framework (VllmEngine for Qwen2.5-VL)
- **[vLLM](https://github.com/vllm-project/vllm)**: High-performance VLM serving
- **[VERL](https://github.com/volcengine/verl)**: RL training framework (modified fork included)
- **[Trimesh](https://trimsh.org/)**: 3D mesh collision detection
- **[Blender](https://www.blender.org/)**: Scene rendering (v4.0.2)
- **[Shapely](https://shapely.readthedocs.io/)**: 2D geometry operations

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📖 Citation

If you find SceneReVis useful in your research, please consider citing:

```bibtex
@article{scenerevis2025,
  title={SceneReVis: Iterative 3D Indoor Scene Generation with Vision-Language Reinforcement Learning},
  author={},
  journal={},
  year={2025}
}
```
