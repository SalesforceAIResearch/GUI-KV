# GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness

[![arXiv](https://img.shields.io/badge/arXiv-2510.00536-b31b1b.svg)](https://arxiv.org/abs/2510.00536)

> Research code accompanying “GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness” (Huang et al., 2025). This repository is provided to support reproducibility and further academic exploration. It is **not** a polished product and will evolve as the paper, datasets, and upstream model releases change.

---

## Repository Layout

```
GUI-KV_GH/
├── gui_kv/                   # Core GUI-KV cache compression utilities
│   └── gui_kv_utils.py
└── eval/                     # Benchmark evaluation pipelines
    ├── *_eval.py             # Python entry points
    ├── *_eval.sh             # Orchestration scripts (preferred entry)
    ├── attention_helpers.py  # Attention overrides and helpers
    ├── process_utils.py      # Shared helpers for data / metrics
    └── ...                   # Benchmark-specific utilities
```

---

## GUI-KV in Brief

- Plug-and-play cache compression that works with existing vision-language GUI agents without retraining.
- Spatial awareness keeps salient regions via hidden-state importance scores.
- Temporal consolidation removes redundant history by projecting past frames onto current key subspaces.
- Achieves substantial FLOP reductions while retaining or improving task success rates across UI automation benchmarks.

See the paper for complete technical details and ablations.

---

## Environment Setup

### Requirements

- Python ≥ 3.10
- Linux with CUDA-enabled GPUs (scripts target 7B-class multimodal models)
- CUDA 12.x (for PyTorch and flash-attention)

### Installation

1. Install transformers from source (required):

   ```bash
   pip install git+https://github.com/huggingface/transformers.git@bbca9782ca1b8b358cc832a1b821aa1b450850da
   ```

   **Note:** The code requires a specific development version of transformers (4.54.0.dev0) from commit `bbca9782`.

2. Install other core dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install flash-attention (if not already installed):

   ```bash
   pip install flash-attn --no-build-isolation
   ```

   **Note:** Flash-attention requires CUDA and may take several minutes to compile. See the [flash-attention repository](https://github.com/Dao-AILab/flash-attention) for troubleshooting.

4. Benchmark datasets:
   - AgentNetBench, AndroidControl, ScreenSpot v2 / Pro, Multimodal Mind2Web
   - See **Dataset Configuration** below for how to configure paths



---

## Dataset Configuration

Before running benchmarks, configure the dataset paths. You can do this in two ways:

### Option 1: Set Environment Variables (Recommended)

Export the required environment variables before running the scripts:

```bash
# AgentNetBench
export AGENTNETBENCH_IMGS=/path/to/AgentNetBench/test_data/images
export AGENTNETBENCH_DATA=/path/to/AgentNetBench/test_data

# AndroidControl
export ANDROIDCONTROL_IMGS=/path/to/AndroidControl/images
export ANDROIDCONTROL_TEST=/path/to/AndroidControl/data

# Multimodal Mind2Web
export MM_MIND2WEB_IMGS=/path/to/Multimodal-Mind2Web/release_images
export MM_MIND2WEB_TEST=/path/to/Multimodal-Mind2Web/data/samples

# ScreenSpot v2
export SCREENSPOT_IMGS=/path/to/ScreenSpot-v2/screenspotv2_image
export SCREENSPOT_TEST=/path/to/ScreenSpot-v2

# ScreenSpot Pro
export SCREENSPOTPRO_IMGS=/path/to/ScreenSpot-Pro/images
export SCREENSPOTPRO_TEST=/path/to/ScreenSpot-Pro/annotations
```

### Option 2: Edit Shell Scripts Directly

Open the desired shell script (e.g., `eval/agentnetbench_eval.sh`) and modify the dataset path defaults near the top of the file.

---

## Running Benchmarks

The recommended entry points are the shell scripts under `eval/`. Each script sweeps KV cache budgets, launches background jobs with `nohup`, and writes logs/results under `logs/` and `results/`.

### Supported Models and KV Cache Methods

All shell scripts support:

**Models:**
- `ByteDance-Seed/UI-TARS-1.5-7B`
- `xlangai/OpenCUA-7B`

**KV Cache Methods:**
- `original` - No compression (full cache)
- `pyramid_kv` - [Pyramid KV](https://arxiv.org/abs/2406.02069)
- `vl_cache` - [VL-Cache](https://arxiv.org/abs/2410.23317)
- `snap_kv` - [SnapKV](https://arxiv.org/abs/2404.14469)
- `gui_kv` - [GUI-KV (ours)](https://arxiv.org/abs/2510.00536)

You can switch models or cache methods by editing the `model_path` and `kv_cache` variables at the top of each shell script.

### Available Benchmarks

| Shell script | Benchmark | Default model | Required environment variables |
|--------------|-----------|---------------|--------------------------------|
| `eval/agentnetbench_eval.sh` | AgentNetBench | `ByteDance-Seed/UI-TARS-1.5-7B` | `AGENTNETBENCH_IMGS`, `AGENTNETBENCH_DATA` |
| `eval/androidcontrol_eval.sh` | AndroidControl | `ByteDance-Seed/UI-TARS-1.5-7B` | `ANDROIDCONTROL_IMGS`, `ANDROIDCONTROL_TEST` |
| `eval/multimodal_mind2web_eval.sh` | Multimodal Mind2Web | `ByteDance-Seed/UI-TARS-1.5-7B` | `MM_MIND2WEB_IMGS`, `MM_MIND2WEB_TEST` |
| `eval/screenspotv2_eval.sh` | ScreenSpot v2 | `ByteDance-Seed/UI-TARS-1.5-7B` | `SCREENSPOT_IMGS`, `SCREENSPOT_TEST` |
| `eval/screenspotpro_eval.sh` | ScreenSpot Pro | `ByteDance-Seed/UI-TARS-1.5-7B` | `SCREENSPOTPRO_IMGS`, `SCREENSPOTPRO_TEST` |

### Step-by-step

1. Configure dataset paths using environment variables (see **Dataset Configuration** above) or by editing the shell script directly.
2. Adjust `kv_cache_budgets`, `num_gpus`, `alpha`, `temperature`, or `window_size` in the shell script if you want to explore other settings.
3. Run the script, for example:

   ```bash
   bash eval/agentnetbench_eval.sh
   ```

   Each budget value is dispatched in the background. Monitor progress with the printed log paths (for example, `tail -f logs/agentnetbench_eval/...`).

4. Aggregated metrics and detailed JSON traces will appear under `results/<benchmark>/...`.

### Running a Single Configuration Manually

If you prefer not to spawn multiple jobs, call the Python entry point directly:

```bash
python eval/agentnetbench_eval.py \
  --model_path ByteDance-Seed/UI-TARS-1.5-7B \
  --agentnetbench_imgs /path/to/AgentNetBench/test_data/images \
  --agentnetbench_data /path/to/AgentNetBench/test_data \
  --kv_cache gui_kv \
  --kv_cache_budget 10 \
  --alpha 2 \
  --window_size 8 \
  --temperature 3.5 \
  --attention_implementation flash_attention_2 \
  --model_dtype bfloat16 \
  --results_dir results/agentnetbench/custom_run \
  --task all
```

All Python drivers share a similar argument set (`--kv_cache`, `--kv_cache_budget`, `--alpha`, `--window_size`, `--temperature`, etc.), so you can adapt the command for other datasets.

---

## Using GUI-KV Programmatically

Advanced users can plug GUI-KV into new experiments via `gui_kv/gui_kv_utils.py`:

```python
from gui_kv.gui_kv_utils import init_gui_kv

# Example hook: register GUI-KV on top of an existing attention module.
kv_config = dict(
    max_capacity_prompt=320,
    window_size=8,
    alpha=2.0,
    temperature=3.5,
    pooling="avgpool",
)
init_gui_kv(model, **kv_config)
```

See the evaluation scripts in `eval/attention_helpers.py` for complete integration examples with Qwen2.5-VL and OpenCUA backends.

---

## Research Use Only

The code is supplied “as is” for academic research. It may rely on proprietary datasets or models that require separate licensing. Always verify you have permission to use any third-party assets referenced by the scripts.

---

## Citation

```bibtex
@article{huang2025guikv,
  title   = {GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness},
  author  = {Huang, Kung-Hsiang and Qiu, Haoyi and Dai, Yutong and Xiong, Caiming and Wu, Chien-Sheng},
  journal = {arXiv preprint arXiv:2510.00536},
  year    = {2025}
}
```

---

## License

Distributed under the terms of the repository’s [LICENSE.txt](LICENSE.txt). Evaluate the license alongside any external model or dataset licenses before use.

---

## Contact

For academic inquiries, reach out to the authors via the contact details in the paper or open an issue in this repository.

