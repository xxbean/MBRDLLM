import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_mbr import (  # noqa: E402
    build_answer_candidates,
    normalize_candidate_weights,
    normalize_gsm8k_answer,
    select_mbr_candidate,
    should_mbr_exit,
)


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        table = {
            1: " 42",
            2: " 42.0",
            3: " 17",
            4: " answer",
        }
        return "".join(table.get(int(token), str(token)) for token in tokens)


def test_normalize_gsm8k_answer():
    assert normalize_gsm8k_answer("#### 42") == "42"
    assert normalize_gsm8k_answer("The answer is 42.") == "42"
    assert normalize_gsm8k_answer("42.0") == "42"
    assert normalize_gsm8k_answer("1,234.00") == "1234"


def test_build_answer_candidates_respects_candidate_budget():
    log_probs = torch.log_softmax(torch.tensor([[[0.0, 4.0, 3.0, 2.0], [0.0, 1.0, 4.0, 3.0]]]), dim=-1)
    candidates = build_answer_candidates(log_probs, [0, 1], token_topk=3, candidate_k=4)
    assert len(candidates) == 4
    assert all(len(tokens) == 2 for tokens, _ in candidates)


def test_normalize_candidate_weights_sum_to_one():
    weighted = normalize_candidate_weights([([1], 0.0), ([2], -1.0), ([3], -2.0)])
    assert math.isclose(sum(weight for _, _, weight in weighted), 1.0, rel_tol=1e-6)


def test_select_mbr_candidate_groups_numeric_answers():
    weighted = normalize_candidate_weights([([1], 1.0), ([2], 0.8), ([3], 0.7)])
    selected = select_mbr_candidate(weighted, tokenizer=FakeTokenizer())
    assert selected["selected_key"] == "42"
    assert selected["candidate_count"] == 3
    assert selected["risk"] < 0.5


def test_should_mbr_exit_phase_thresholds():
    thresholds = {"early": 0.05, "mid": 0.10, "late": 0.20}
    assert should_mbr_exit(10, 100, 0.04, thresholds)
    assert not should_mbr_exit(10, 100, 0.06, thresholds)
    assert should_mbr_exit(50, 100, 0.09, thresholds)
    assert not should_mbr_exit(50, 100, 0.12, thresholds)
    assert should_mbr_exit(90, 100, 0.19, thresholds)


if __name__ == "__main__":
    test_normalize_gsm8k_answer()
    test_build_answer_candidates_respects_candidate_budget()
    test_normalize_candidate_weights_sum_to_one()
    test_select_mbr_candidate_groups_numeric_answers()
    test_should_mbr_exit_phase_thresholds()
    print("All MBMBR helper tests passed.")
