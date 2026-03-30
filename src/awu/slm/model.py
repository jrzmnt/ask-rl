from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------------------------------
# Output container
# -------------------------------------------------

@dataclass
class SLMOutput:
    text: str
    logits: torch.Tensor
    cost: float


# -------------------------------------------------
# SLM wrapper
# -------------------------------------------------

class HuggingFaceSLM:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
    ):
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

        self.model.eval()

    def generate(self, prompt: str, decoding: Dict[str, Any]) -> SLMOutput:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=decoding["max_tokens"],
                # temperature=decoding.get("temperature", 1.0),
                # top_p=decoding.get("top_p", 1.0),
                # do_sample=decoding.get("temperature", 0.0) > 0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # generated text
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        # logits of first generated token
        logits = outputs.scores[0][0]

        # very rough cost proxy (token count)
        cost = generated_ids.numel()

        return SLMOutput(
            text=text,
            logits=logits,
            cost=float(cost),
        )


# -------------------------------------------------
# Factory
# -------------------------------------------------

def load_slm(cfg: Dict[str, Any]) -> HuggingFaceSLM:
    provider = cfg.get("provider", "hf")

    if provider != "hf":
        raise ValueError(f"Unsupported SLM provider: {provider}")

    return HuggingFaceSLM(
        model_name=cfg["model"],
        device=cfg.get("device", "auto"),
        dtype=cfg.get("dtype", "float16"),
    )
