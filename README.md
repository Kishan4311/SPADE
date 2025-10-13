# SPADE — Speculative Decoding for Precise & Low‑Cost Distributed Edge‑Cloud Inference

**SPADE-Speculative-Decoding-for-Precise-and-Low-Cost-Distributed-Edge-Cloud-Inference**

SPADE is a distributed inference framework that combines speculative decoding across edge and cloud to deliver accurate, low-cost generation without retraining. A compact *drafter* on the edge proposes candidate tokens while a larger *verifier* in the cloud validates them in parallel. Accepted tokens are kept locally; only rejected tokens trigger the cloud verifier — reducing cloud queries and cost while keeping model output quality.

---

## Key features

* Implements Speculative Decoding adapted for distributed edge ↔ cloud inference.
* Interactive UI to run prompts and debug (run `inference.py`).
* Supports LLaMA-family models (configurable target/drafter models).
* Evaluation suites for **CNN / DailyMail** and **SPEC‑BENCH** datasets across multiple tasks.
* Modular, easy-to-run scripts for reproducing experiments and extending SPADE.
* Fully documented usage and ready for research or production prototyping.

---

## Quick start

### Clone the repository

```bash
git clone https://github.com/Kishan4311/SPADE-Speculative-Decoding-for-Precise-and-Low-Cost-Distributed-Edge-Cloud-Inference.git
cd SPADE-Speculative-Decoding-for-Precise-and-Low-Cost-Distributed-Edge-Cloud-Inference
```

### Install requirements

```bash
pip install -r requirements.txt
```

### Hugging Face authentication

```bash
huggingface-cli login
# or
from huggingface_hub import login
login("<YOUR_HF_TOKEN>")
```

---

## Run interactive chat (edge-side drafter)

Run the interactive inference UI (uses the drafter on the specified device):

```bash
python inference.py --device "cuda:0"
```

You can swap `--device` to `cpu` or another CUDA device as needed.

---

## Evaluation

> Note: for evaluation, choose larger target (verifier) models such as `meta-llama/Llama-3.1-8B-Instruct` (or newer) and set output paths accordingly. The examples below show typical invocation patterns.

### Evaluate SPEC‑BENCH dataset

```bash
python evaluation/evaluate_specbench.py \
  --code-path "inference.py" \
  --specbench "data/SpecBench_question.jsonl" \
  --out "OUTPUTPATH/specbench_results_gm8_tgt8B_genLen256.jsonl" \
  --summary "OUTPUTPATH/specbench_summary.json" \
  --device "cuda:0" \
  --gamma 6 \
  --max-gen-len 256 \
  --max-examples 480 \
  --target-model "meta-llama/Llama-3.2-3B-Instruct" \
  --drafter-model "meta-llama/Llama-3.2-1B-Instruct"
```

### Evaluate CNN / DailyMail dataset

```bash
python evaluation/evaluate_cnnDailyMail.py \
  --device "cuda:0" \
  --dataset "data/cnnDailyMail.json" \
  --specdec_path "inference.py" \
  --output "OUTPUTPATH/cnnDailyMail_result_gm6_tgt3B_genLen128.json" \
  --k 200 \
  --gen_len 128 \
  --gamma 6 \
  --target_model "meta-llama/Llama-3.2-3B-Instruct" \
  --drafter_model "meta-llama/Llama-3.2-1B-Instruct"
```

> Tip: adjust `--gamma` (speculative length), `--gen_len` / `--max-gen-len`, and model choices to balance cost vs. accuracy. Increase `--max-examples` or `--k` for larger evaluation runs.

---

## Configuration & models

* Edit model names and device options directly in the script invocation or in the config file (if present).
* Make sure the drafter is a smaller, fast model and the target/verifier is a larger high-quality model for best cost/accuracy tradeoffs.

---

## Reproducing experiments

* All experiments are scripted under `evaluation/` — modify paths and flags to match your environment.
* Keep consistent HF model identifiers and ensure you have permission/access to the models used.

---

## Contributing

Contributions, issues and feature requests are welcome. Please open an issue or submit a pull request. For major changes, open an issue first to discuss the design.

---

## License

This project is provided under the terms of the <>. See `LICENSE` for details.

---

## Citation

If you use SPADE in your research, please consider citing the repository and include a brief note in your methods describing the drafter/verifier split and the speculative decoding setup.

---

## Contact

For questions or collaboration: open an issue on GitHub or reach out to the repository owner.
