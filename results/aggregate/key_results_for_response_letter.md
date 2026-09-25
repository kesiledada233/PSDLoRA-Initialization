# Key Results for Response Letter

No formal matrix result is available yet. Formal task/convergence/mechanism tables must only be populated from completed, schema-valid run IDs; Gate diagnostics and engineering smokes remain separately labeled below.

Current engineering evidence:

- CPU initializer/schema/PSD/sandbox/checkpoint/gate suite: 143/143 unit tests and 33/33 executable CLI help checks passed from clean commit `0e54dc78b27e525e00d810cd07919f95a5631ff5` on 2026-09-05; canonical report SHA256 `118b59887fc2b7a4339cd920abfe6c9f787dc84793b75e1add014ad7042b3b37`.
- Legacy Gate 1-R provenance: 13-file openPangu/GSM8K PEFT-default bundle across seeds 1107/123/42 independently hash-verified. The corrected CUDA reconstruction produced AUC500 3131.68 within the submitted range 3095.26–3160.08 and step-500 loss 0.3897 within 0.3259–0.4766. Its exact original per-run `config.json`, argv, Git and environment remain unavailable, so this is explicitly a provenance gate and cannot be reused as a revision result.
- Gate 2 loss diagnosis: on the same 100 openPangu/GSM8K batches, correct padding masking gives median loss 2.8346 while the submitted unmasked-padding policy gives 16.9242 (increase 14.0896); PEFT-default and FDA both have zero nonzero B entries and exact zero logits/loss deviation from the base model at step zero. The independent Qwen/CMMLU 100-batch case also has exact step-zero equivalence.
- Integration execution evidence: the isolated PEFT/MBPP adapter-and-executor path, official 8-batch LoRA-One path (56 initialized modules), and 20-step gradient logger (21 recorded steps) all passed their hash-bound audit. MBPP smoke pass@1=0/3 is not a formal metric and must not be cited as model-quality evidence.
- openPangu checkpoint: pinned revision and 4 LFS shards / 479 indexed tensor headers verified.
- Qwen2.5-7B checkpoint: pinned revision and 4 LFS shards / 339 indexed tensor headers verified.
- Gate 0: both 7B checkpoints loaded in BF16 and produced finite logits on RTX 5090; this is engineering readiness only, not a scientific result.
- Prometheus 2 judge: pinned 8-shard checkpoint / 291 tensor headers verified; official-format 3-case smoke produced parseable scores and ranked a correct arithmetic answer (4) above an incorrect one (1).
- openPangu scope contract: 238 exact all-linear target paths; rank-16 trainable parameter counts are 7,241,728 for q/v and 42,057,728 for all-linear.
- CMMLU processed split: 335 train / 11,582 test.
- MBPP processed split: 600 train / 374 test.
- ShareGPT frozen split: 18,623 train / 2,069 test.
- ShareGPT judge prompt set: 200 fixed test prompts; SHA256 `b85da6f14b6047653f13fb0d5d407acc46e28aba429bba76c6c4fca333a24099`.

Only the explicitly labeled Gate 2 same-batch diagnostic supports the legacy-loss explanation. Do not present Gate 0, integration smokes, inventories, or CPU tests as formal matrix evidence or as resolution of other reviewer concerns.
