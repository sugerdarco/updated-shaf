"""Chat-templated Stage-8 agent.

The SAHF byte loop hands every agent the same running string
``user_prompt + assistant_so_far`` and lets each re-encode it with its own
tokenizer. Instruct models degrade badly when fed a bare prompt with no chat
template, which starves the fusion of good inputs. This wrapper re-encodes as

    <model chat template>(user_prompt)  +  assistant_so_far

so each model sees its native instruct format while still sharing the one
consensus continuation the ensemble is building.
"""
from __future__ import annotations

import numpy as np

from pipeline.stage_0_agent_ensemble import DirectHFAgent


class ChatTemplateAgent(DirectHFAgent):
    def set_prompt(self, prompt: str) -> "ChatTemplateAgent":
        """Fix the user turn for the current generation; call once per prompt."""
        self._user_prompt = prompt
        self._tmpl_ids = None
        self._cache.clear()
        self._stop_cache.clear()
        return self

    def _templated_ids(self):
        import torch  # noqa: F401  (kept lazy like the base class)

        if getattr(self, "_tmpl_ids", None) is None:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": self._user_prompt}],
                add_generation_prompt=True, return_tensors="pt",
            )
            self._tmpl_ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return self._tmpl_ids

    def next_token_probs(self, context: str) -> np.ndarray:
        up = getattr(self, "_user_prompt", None)
        if up is None or not context.startswith(up):
            return super().next_token_probs(context)  # fallback: bare encoding
        if context in self._cache:
            return self._cache[context]

        import torch

        base = self._templated_ids()
        assistant = context[len(up):]
        if assistant:
            cont = self.tokenizer(assistant, return_tensors="pt",
                                  add_special_tokens=False).input_ids
            input_ids = torch.cat([base, cont], dim=1)
        else:
            input_ids = base
        if self.device:
            input_ids = input_ids.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1].float()
        if self.temperature != 1.0:
            logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)

        out = probs.detach().cpu().numpy().astype(np.float64)
        n = len(self._vocab.token_bytes)
        if out.shape[0] < n:
            out = np.pad(out, (0, n - out.shape[0]))
        elif out.shape[0] > n:
            out = out[:n]
        out = out.copy()

        stop_ids = {i for i in self._eos_ids if i < out.shape[0]}
        self._stop_cache[context] = float(out[list(stop_ids)].sum()) if stop_ids else 0.0
        for sid in self._vocab.special_ids:
            if sid < out.shape[0]:
                out[sid] = 0.0
        s = out.sum()
        if s > 0:
            out /= s
        self._cache[context] = out
        return out
