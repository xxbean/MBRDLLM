'''LLaDA evaluation harness with Prophet early exit support.'''
import accelerate
import json
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
import time
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from generate_earlyexit import generate as generate_prophet


def _parse_constraints(text: str, tokenizer) -> dict[int, int]:
    """Parse constraint string like "120:THE|121:ANSWER" into position->token_id dict."""
    constraints: dict[int, int] = {}
    if text is None or text.strip() == "":
        return constraints
    for part in text.split('|'):
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
        except ValueError:
            continue
        word = word.strip()
        # Prepend space for tokenization consistency
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, tid in enumerate(ids):
            constraints[pos + i] = tid
    return constraints


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        **kwargs,
    ):
        '''Initialize LLaDA evaluation with optional Prophet early exit.'''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking    
        
        # Early exit and constraint args
        self.constraints_text = kwargs.pop('constraints_text', '')
        self.answer_length = int(kwargs.pop('answer_length', 5))
        self.enable_early_exit = self._as_bool(kwargs.pop('enable_early_exit', False))
        self.early_exit_method = str(kwargs.pop('early_exit_method', '')).strip().strip("'\"").lower()
        if not self.early_exit_method:
            self.early_exit_method = 'prophet' if self.enable_early_exit else 'none'
        if self.early_exit_method not in ('none', 'prophet', 'mbr'):
            raise ValueError(f"Unsupported early_exit_method: {self.early_exit_method}")
        
        # Early exit thresholds (optional)
        self.early_threshold = float(kwargs.pop('early_threshold', 7.5))
        self.mid_threshold = float(kwargs.pop('mid_threshold', 5.0))
        self.late_threshold = float(kwargs.pop('late_threshold', 2.5))

        # MBMBR risk stopping parameters
        self.mbr_candidate_k = int(kwargs.pop('mbr_candidate_k', 8))
        self.mbr_token_topk = int(kwargs.pop('mbr_token_topk', 3))
        self.mbr_risk_early = float(kwargs.pop('mbr_risk_early', 0.05))
        self.mbr_risk_mid = float(kwargs.pop('mbr_risk_mid', 0.10))
        self.mbr_risk_late = float(kwargs.pop('mbr_risk_late', 0.20))
        self.metrics_log_path = str(kwargs.pop('metrics_log_path', '')).strip().strip("'\"")
        
    def _as_bool(self, v, default=False):
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().strip("'\"").lower()
        return s in ("1", "true", "yes", "y", "t")

    def _append_metrics(self, record):
        if not self.metrics_log_path:
            return
        if hasattr(self, '_rank') and self.rank != 0:
            return
        path = Path(self.metrics_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
 
            # Prepare constraints if provided
            constraints = _parse_constraints(self.constraints_text, self.tokenizer) if self.constraints_text else None
            
            # Determine answer start position for early exit
            answer_start_pos = None
            if self.early_exit_method != 'none' and constraints:
                answer_start = max(constraints.keys()) + 2 if constraints else 0
                answer_start_pos = prompt.shape[1] + answer_start
 
            generation_info = {
                'early_exit_triggered': False,
                'exit_decision_step': None,
                'total_steps': self.steps,
                'actual_steps': self.steps,
            }

            # Generate with full-step, Prophet, or MBMBR early stopping.
            if self.early_exit_method == 'prophet':
                generated_out, gap_data = generate_prophet(
                    self.model,
                    prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    constraints=constraints,
                    analyze_gap=True,
                    answer_start_pos=answer_start_pos,
                    tokenizer=self.tokenizer,
                    early_exit_thresholds={
                        'early': self.early_threshold,
                        'mid': self.mid_threshold,
                        'late': self.late_threshold
                    },
                    measure_time=True,
                )
                generation_info.update(gap_data.get('exit_info', {}))
            elif self.early_exit_method == 'mbr':
                from generate_mbr import generate as generate_mbr
                generated_out, gap_data = generate_mbr(
                    self.model,
                    prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    constraints=constraints,
                    analyze_gap=True,
                    answer_start_pos=answer_start_pos,
                    answer_length=self.answer_length,
                    tokenizer=self.tokenizer,
                    mbr_candidate_k=self.mbr_candidate_k,
                    mbr_token_topk=self.mbr_token_topk,
                    mbr_risk_thresholds={
                        'early': self.mbr_risk_early,
                        'mid': self.mbr_risk_mid,
                        'late': self.mbr_risk_late,
                    },
                    measure_time=True,
                )
                generation_info.update(gap_data.get('exit_info', {}))
            else:
                from generate import generate as generate_baseline
                start_time = time.time()
                generated_out = generate_baseline(
                    self.model,
                    prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    constraints=constraints,
                )
                generation_info['inference_time'] = time.time() - start_time
 
            generated_answer = self.tokenizer.decode(generated_out[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # Remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            self._append_metrics({
                'method': self.early_exit_method,
                'actual_steps': generation_info.get('actual_steps', self.steps),
                'total_steps': generation_info.get('total_steps', self.steps),
                'risk': generation_info.get('risk'),
                'mbr_score': generation_info.get('mbr_score'),
                'selected_answer': generation_info.get('selected_answer', ''),
                'selected_key': generation_info.get('selected_key', ''),
                'candidate_count': generation_info.get('candidate_count', 0),
                'early_exit_triggered': generation_info.get('early_exit_triggered', False),
                'exit_decision_step': generation_info.get('exit_decision_step'),
                'inference_time': generation_info.get('inference_time'),
                'generated_answer': generated_answer,
            })

            # Log input & output
            if not hasattr(self, '_rank') or self.rank == 0:
                question_text = elem.get("question_text", "<N/A>")
                print(f"[LOG][Prompt] {question_text}")
                print(f"[LOG][Answer] {generated_answer}\n")

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
