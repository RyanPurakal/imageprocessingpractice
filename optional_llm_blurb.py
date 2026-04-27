#!/usr/bin/env python3
"""
Standalone text-generation demo using DistilGPT-2 via the Hugging Face `pipeline` API; has no dependency on the image denoising pipeline and is included solely as a lightweight LLM illustration.
Requires `transformers` (already in requirements.txt); downloads the model on first run.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prompt",
        type=str,
        default="Scientific image restoration for simulation data is important because",
    )
    p.add_argument("--max-new-tokens", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import pipeline

    gen = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    out = gen(args.prompt, num_return_sequences=1)[0]["generated_text"]
    print(out)


if __name__ == "__main__":
    main()
