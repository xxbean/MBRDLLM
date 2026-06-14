import re
import time

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """Gumbel-Max sampling with float64 precision."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Calculate tokens to transfer at each step for uniform denoising."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def normalize_gsm8k_answer(text):
    """Return the final numeric answer string used for GSM8K-style agreement."""
    if text is None:
        return ""
    text = str(text)
    if "####" in text:
        text = text.split("####")[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return ""
    answer = numbers[-1].replace(",", "")
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".")
    return answer


def normalize_fallback_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def build_answer_candidates(log_probs, answer_positions, token_topk=3, candidate_k=8):
    """Build S_n from per-position top-k token log probabilities."""
    if not answer_positions:
        return []

    beams = [([], 0.0)]
    for pos in answer_positions:
        if pos < 0 or pos >= log_probs.shape[1]:
            continue
        k = min(token_topk, log_probs.shape[-1])
        values, token_ids = torch.topk(log_probs[0, pos], k=k, dim=-1)
        next_beams = []
        for prefix, prefix_score in beams:
            for value, token_id in zip(values.tolist(), token_ids.tolist()):
                next_beams.append((prefix + [int(token_id)], float(prefix_score + value)))
        next_beams.sort(key=lambda item: item[1], reverse=True)
        beams = next_beams[:candidate_k]

    return beams[:candidate_k]


def normalize_candidate_weights(candidates):
    """Convert candidate log scores into normalized model-based weights."""
    if not candidates:
        return []
    scores = torch.tensor([score for _, score in candidates], dtype=torch.float32)
    weights = torch.softmax(scores, dim=0).tolist()
    return [(tokens, score, float(weight)) for (tokens, score), weight in zip(candidates, weights)]


def select_mbr_candidate(weighted_candidates, tokenizer=None):
    """Select y_hat_n by maximum expected numeric-answer agreement."""
    if not weighted_candidates:
        return {
            "selected_tokens": [],
            "selected_text": "",
            "selected_key": "",
            "risk": 1.0,
            "mbr_score": 0.0,
            "candidate_count": 0,
        }

    decoded = []
    for tokens, score, weight in weighted_candidates:
        if tokenizer is None:
            text = " ".join(str(token) for token in tokens)
        else:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        numeric_key = normalize_gsm8k_answer(text)
        key = numeric_key if numeric_key else normalize_fallback_text(text)
        decoded.append(
            {
                "tokens": tokens,
                "score": score,
                "weight": weight,
                "text": text,
                "key": key,
            }
        )

    group_mass = {}
    for item in decoded:
        group_mass[item["key"]] = group_mass.get(item["key"], 0.0) + item["weight"]

    selected_key, mbr_score = max(group_mass.items(), key=lambda item: item[1])
    group_items = [item for item in decoded if item["key"] == selected_key]
    selected = max(group_items, key=lambda item: item["score"])
    risk = 1.0 - float(mbr_score)
    return {
        "selected_tokens": selected["tokens"],
        "selected_text": selected["text"],
        "selected_key": selected_key,
        "risk": risk,
        "mbr_score": float(mbr_score),
        "candidate_count": len(decoded),
    }


def should_mbr_exit(current_step, max_steps, risk, thresholds=None):
    """Phase-aware MBMBR risk stopping rule."""
    if risk is None:
        return False
    if thresholds is None:
        thresholds = {"early": 0.05, "mid": 0.10, "late": 0.20}

    progress = current_step / max_steps
    if progress < 0.33:
        return risk <= thresholds.get("early", 0.05)
    if progress < 0.67:
        return risk <= thresholds.get("mid", 0.10)
    return risk <= thresholds.get("late", 0.20)


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking="low_confidence", mask_id=126336, constraints=None,
             analyze_gap=False, tokenizer=None, answer_start_pos=None, answer_length=5,
             mbr_candidate_k=8, mbr_token_topk=3, mbr_risk_thresholds=None,
             measure_time=False, **_):
    """LLaDA generation with stage-wise model-based MBR early stopping."""

    early_exit_triggered = False
    exit_decision_step = None
    latest_mbr = None
    inference_start_time = time.time() if measure_time else None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if constraints is not None:
        for pos, token_id in constraints.items():
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0
    max_steps = steps

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            global_step += 1
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            if answer_start_pos is not None:
                answer_positions = list(
                    range(answer_start_pos, min(prompt.shape[1] + gen_length, answer_start_pos + answer_length))
                )
                log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
                candidates = build_answer_candidates(
                    log_probs,
                    answer_positions,
                    token_topk=mbr_token_topk,
                    candidate_k=mbr_candidate_k,
                )
                weighted_candidates = normalize_candidate_weights(candidates)
                latest_mbr = select_mbr_candidate(weighted_candidates, tokenizer=tokenizer)

                if not early_exit_triggered and should_mbr_exit(
                    global_step,
                    max_steps,
                    latest_mbr["risk"],
                    mbr_risk_thresholds,
                ):
                    print(
                        f"MBR early exit at step {global_step}/{max_steps} "
                        f"with risk={latest_mbr['risk']:.3f}"
                    )
                    exit_decision_step = global_step
                    early_exit_triggered = True

                    remaining_mask = (x == mask_id)
                    x[remaining_mask] = x0[remaining_mask]
                    for offset, token_id in enumerate(latest_mbr["selected_tokens"]):
                        absolute_pos = answer_start_pos + offset
                        if absolute_pos < x.shape[1]:
                            x[:, absolute_pos] = int(token_id)
                    break

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

            if constraints is not None:
                for pos, token_id in constraints.items():
                    absolute_pos = prompt.shape[1] + pos
                    if absolute_pos < x.shape[1]:
                        x[:, absolute_pos] = token_id

        if early_exit_triggered:
            break

    if analyze_gap:
        exit_info = {
            "early_exit_triggered": early_exit_triggered,
            "exit_decision_step": exit_decision_step,
            "total_steps": max_steps,
            "actual_steps": exit_decision_step if early_exit_triggered else max_steps,
            "risk": latest_mbr["risk"] if latest_mbr else None,
            "mbr_score": latest_mbr["mbr_score"] if latest_mbr else None,
            "selected_answer": latest_mbr["selected_text"] if latest_mbr else "",
            "selected_key": latest_mbr["selected_key"] if latest_mbr else "",
            "candidate_count": latest_mbr["candidate_count"] if latest_mbr else 0,
        }
        if measure_time:
            exit_info["inference_time"] = time.time() - inference_start_time
        return x, {"exit_info": exit_info}

    return x
