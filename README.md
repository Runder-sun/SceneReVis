# SceneReVis

**SceneReVis: Iterative 3D Indoor Scene Generation with Vision-Language Reinforcement Learning**

A closed-loop framework for generating physically plausible and aesthetically coherent 3D indoor scenes through multi-turn iterative refinement. The system combines Large Language Model (LLM) reasoning, Vision-Language Model (VLM) feedback, and physics-based validation to produce high-quality 3D room layouts.

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
├── infer.py                      # Single-GPU inference with iterative refinement
├── infer_batch.py                # Multi-GPU batch inference
├── infer_amlt.py                 # Cloud (AMLT) inference variant
├── optimize_scene.py             # Scene physics optimization
├── batch_optimize_integrated.py  # Batch scene optimization
├── batch_optimize_rooms.py       # Room-level batch optimization
│
├── eval/                         # Evaluation tools
│   ├── myeval.py                 # Mesh-based collision & OOB evaluation
│   ├── myeval_bbox.py            # Bounding box-based evaluation
│   ├── myeval_mesh_notol.py      # Mesh evaluation (no tolerance)
│   ├── voxel_eval.py             # Voxel-based spatial evaluation
│   ├── vlm_scene_eval.py         # VLM (GPT-4o Vision) multi-dimension eval
│   ├── calculate_collision_oob.py # Collision/OOB rate calculation
│   ├── batch_eval_all_baselines.py # Batch baseline evaluation
│   └── batch_vlm_eval.py         # Batch VLM evaluation
│
├── utils/                        # Core utilities
│   ├── sample.py                 # 3D-FUTURE asset retrieval (SigLIP-based)
│   ├── objaverse_retriever.py    # Objaverse asset retrieval (CLIP+SBERT)
│   ├── objaverse_glb_manager.py  # Objaverse GLB asset management
│   ├── scene_editor.py           # Scene editing operations (add/remove/move/etc.)
│   ├── format_converter.py       # Scene format conversion (flat ↔ grouped)
│   ├── blender_renderer.py       # Blender rendering engine
│   ├── blender_wrapper.py        # Blender subprocess wrapper
│   ├── main_bpy.py               # Blender script entry point
│   ├── visualization_3d.py       # 3D visualization (bbox, arrows, grid)
│   ├── RL_utils.py               # RL training utilities
│   ├── path_config.py            # Unified path configuration manager
│   └── image_merger.py           # Multi-view image composition
│
├── dataprocess/                  # Data processing pipeline
│   ├── generate_final_conversations_v3.py  # CoT conversation generation
│   ├── prepare_rl_data.py        # RL training data preparation
│   ├── generate_intermediate_data.py       # Intermediate data generation
│   ├── batch_process_scenes.py   # Batch scene processing
│   ├── evaluate_and_filter_chains.py       # Chain quality filtering
│   ├── validate_scenes_voxel.py  # Voxel-based scene validation
│   └── ...
│
├── script/                       # Training scripts
│   ├── RL/                       # Reinforcement learning
│   │   ├── scene_reward.py       # Reward function
│   │   ├── scene_editing_interaction.py  # RL interaction handler
│   │   ├── run_grpo_B200.sh      # GRPO training launch script
│   │   └── config/               # RL configuration files
│   └── sft/                      # Supervised fine-tuning
│       └── sft_B200.sh           # SFT training launch script
│
├── verl/                         # VERL RL framework (modified)
│   └── verl/
│       ├── interactions/         # Multi-turn interaction interfaces
│       │   ├── base.py           # Base interaction class
│       │   └── scene_editing_interaction.py  # Scene editing interaction
│       ├── trainer/              # Training orchestration
│       └── ...
│
├── metadata/                     # Asset metadata
│   ├── model_info_3dfuture_assets.json
│   └── invalid_threed_front_rooms.txt
│
├── requirements_infer_batch.txt  # Inference dependencies
├── setup_env.sh                  # Environment setup
└── quick_install_blender.sh      # Blender installation
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n scenerevis python=3.11 -y
conda activate scenerevis

# Install dependencies
pip install -r requirements_infer_batch.txt

# Install Blender for rendering
bash quick_install_blender.sh

# Install VERL framework (for RL training)
cd verl && pip install -e . && cd ..
```

### 2. Download Required Assets

You need to download the following assets separately:

- **3D-FUTURE models**: Download from [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) and place under your datasets directory
- **Objaverse assets** (optional): For extended object library
- **Metadata files**: Download embeddings pickle file (not included due to size)

### 3. Configuration

Update paths in your environment or config files:

```bash
# Set environment variables
export PTH_3DFUTURE_ASSETS=/path/to/3D-FUTURE-model
export PTH_ASSETS_METADATA=./metadata/model_info_3dfuture_assets.json
export PTH_ASSETS_EMBED=./metadata/model_info_3dfuture_assets_embeds.pickle

# For Azure OpenAI (optional, for VLM feedback)
export AZURE_OPENAI_ENDPOINT=your_endpoint
export AZURE_OPENAI_API_KEY=your_key
```

### 4. Inference

```bash
# Single scene inference
python infer.py --prompt "Design a cozy bedroom with a queen bed and reading corner"

# Batch inference (multi-GPU)
python infer_batch.py --batch-mode --parallel \
    --model /path/to/checkpoint \
    --prompts-file prompts.txt \
    --iterations 10 \
    --output ./output/results
```

### 5. Evaluation

```bash
# Mesh-based physical evaluation
python eval/myeval.py --scene-dir ./output/results --models-path /path/to/3D-FUTURE-model

# Voxel-based evaluation
python eval/voxel_eval.py --scene-dir ./output/results --models-path /path/to/3D-FUTURE-model

# VLM evaluation (requires Azure OpenAI)
python eval/vlm_scene_eval.py --render-dir ./output/rendered --prompts-file prompts.txt
```

### 6. Training

#### SFT (Supervised Fine-Tuning)

```bash
# Prepare SFT data
python dataprocess/generate_final_conversations_v3.py

# Run SFT
bash script/sft/sft_B200.sh
```

#### RL (Reinforcement Learning with GRPO)

```bash
# Prepare RL data
python dataprocess/prepare_rl_data.py

# Run GRPO training
bash script/RL/run_grpo_B200.sh
```

---

## 📊 Evaluation Metrics

| Metric | Description | Tool |
|--------|-------------|------|
| Collision Rate | % of objects with physical overlaps | `myeval.py` / `voxel_eval.py` |
| Out-of-Bounds Rate | % of objects outside room boundaries | `myeval.py` / `voxel_eval.py` |
| VLM Rationality | Scene rationality score (0-100) | `vlm_scene_eval.py` |
| VLM Spatial Layout | Layout quality score (0-100) | `vlm_scene_eval.py` |
| VLM Accessibility | Accessibility score (0-100) | `vlm_scene_eval.py` |

---

## 🔧 Key Dependencies

- **[ms-swift](https://github.com/modelscope/ms-swift)**: Model inference framework
- **[vLLM](https://github.com/vllm-project/vllm)**: High-performance LLM serving
- **[VERL](https://github.com/volcengine/verl)**: RL training framework (modified fork included)
- **[Trimesh](https://trimsh.org/)**: 3D mesh processing
- **[Blender](https://www.blender.org/)**: Scene rendering (v3.6+)
- **[Shapely](https://shapely.readthedocs.io/)**: 2D geometry operations

---

## 📄 License

This project is released under the MIT License.

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
