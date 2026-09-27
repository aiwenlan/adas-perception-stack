# DriveLM

date: prior demo
machine / GPU: RTX 4090 24GB
upstream URL: https://github.com/aiwenlan/Finetune-e2e-model-DriveLM (adapter / demo path used in this project; upstream DriveLM https://github.com/OpenDriveLab/DriveLM)
upstream commit: `4f768aa72ea1dd362b2178e02f117264bc04286d`
checkpoint: official BIAS-7B + LLaMA-1 (not personal finetune weights)
reported metrics: **10/10 non-empty generations** — not QA accuracy
visualization: `assets/drivelm/drivelm_qa_card.png`
sample outputs: `test_output_llama1.json` / `test_llama_10.json` (in this folder)
known limitations: evidence level C; answers need human fact-check for hallucination (see `factcheck_checklist.md`)
