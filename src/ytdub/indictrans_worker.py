"""One-shot IndicTrans2 worker for Hindi -> English translation."""

from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor


MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

LANGUAGE_CODES = {
    "hi": "hin_Deva",
}


def main() -> int:
    payload = json.load(sys.stdin)

    source_language = (
        str(payload["source_language"]).lower().split("-")[0]
    )
    texts = payload["texts"]

    if source_language not in LANGUAGE_CODES:
        raise RuntimeError(
            f"Unsupported IndicTrans2 source language: {source_language}"
        )

    if not isinstance(texts, list) or not texts:
        raise RuntimeError("texts must be a non-empty list")

    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise RuntimeError("all texts must be non-empty strings")

    src_lang = LANGUAGE_CODES[source_language]
    tgt_lang = "eng_Latn"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    processor = IndicProcessor(inference=True)

    batch = processor.preprocess_batch(
        texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    translations = processor.postprocess_batch(
        decoded,
        lang=tgt_lang,
    )

    if len(translations) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} translations, got {len(translations)}"
        )

    sys.stdout.write(
        json.dumps(
            {"translations": translations},
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())