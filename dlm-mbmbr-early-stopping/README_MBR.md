# DLM + MBMBR Early Stopping on GSM8K

This branch adds a GSM8K-focused implementation of the slide proposal:
replace Prophet's top-2 confidence-gap stopping signal with a stage-wise
model-based MBR risk signal.

## Experiment Target

- Benchmark: GSM8K (`gsm8k_cot_zeroshot`)
- Model: `GSAI-ML/LLaDA-8B-Instruct`
- Baselines: full-step LLaDA and Prophet
- Ours: MBMBR-risk early stopping
- Table 1 reference from *Diffusion Language Models Know the Answer Before
  Decoding*: GSM8K LLaDA full-step 77.1%, Prophet 77.9%, speedup 1.63x.

## Slide Symbols to Code

- `R -> S_n`: Paper 2's reference/support set becomes the step-`n` answer
  candidate set built from top-k answer-region logits.
- `P(y) -> P_tilde_n(y | H_n)`: Paper 2's model probability becomes the
  DLM step belief score, implemented as the sum of answer-token log
  probabilities from the current logits.
- `P_hat_{n,MB}`: the candidate scores are softmax-normalized inside `S_n`,
  giving model-based MBR weights.
- `h^MB -> y_hat_n`: the selected model-based MBR hypothesis becomes the
  representative answer at step `n`.

In v1, the utility is GSM8K numeric answer agreement:

```text
u(y, y_hat) = 1 if normalized final numbers match else 0
risk(n) = 1 - max_y_hat sum_y P_hat_{n,MB}(y) u(y, y_hat)
```

This is intentionally lighter than BERTScore or embedding similarity because
GSM8K evaluation is answer-number based and the goal is to test the stopping
signal, not add a heavy semantic scorer.

## Running

Full GSM8K:

```bash
bash eval_mbr_gsm8k.sh
```

Small smoke test:

```bash
LIMIT_ARG="--limit 5" bash eval_mbr_gsm8k.sh
```

Outputs are written to:

- `logs/gsm8k_full.jsonl`
- `logs/gsm8k_prophet.jsonl`
- `logs/gsm8k_mbr.jsonl`

Each JSONL row records `method`, `actual_steps`, `total_steps`, `risk`,
`selected_answer`, `candidate_count`, `inference_time`, and the final generated
answer.

## Reporting

Report a compact table with:

- Accuracy
- Average steps
- Speedup = `256 / avg_steps`
- Average wall-clock time
- Early stop rate
- Prophet stop step vs MBMBR risk stop step
