#!/usr/bin/env python3
"""
Predict Track 1 SAQ only for missing IDs (and optionally re-predict <NO_ANSWER> rows).

Example:
  python predict_missing_saq.py \
    --input "track_1_saq_input 2.tsv" \
    --pred_in "track_1_saq_prediction_shiran_prompt_v1_without_local_db_evaluation_test.tsv" \
    --pred_out "track_1_saq_prediction_filled_all.tsv" \
    --redo_no_answer \
    --model mistral \
    --sleep 0.1
"""

import argparse
import os
import re
import time
import pandas as pd
import ollama


NO_ANSWER = "<NO_ANSWER>"

SYSTEM_STRICT = """You are a factual multilingual assistant for a question-answering benchmark.
Your goal is to give one short, correct answer for each question in its original language.
Instructions:
- Read the question carefully.
- Respond **only with the concise answer** — a word, number, name, or short phrase.
- Do not include explanations, reasoning, or extra words.
- Keep the answer in the **same language** as the question (Chinese → Chinese, English → English, Spanish → Spanish).
- If you are not confident you know the answer from your own knowledge, reply exactly: <NO_ANSWER>
Example outputs:
Q (English): What is the capital of France?
A: Paris
Q (Chinese): 中国的国庆节是哪一天？
A: 10月1日
Q (Spanish): ¿Cuál es el idioma oficial de España?
A: Español
Now answer the following question:
"""

# Second pass: force an answer (because you said you MUST answer all questions)
SYSTEM_FORCE = """You are a factual multilingual assistant for a question-answering benchmark.
Your goal is to give one short, correct answer for each question in its original language.
Instructions:
- Read the question carefully.
- Respond **only with the concise answer** — a word, number, name, or short phrase.
- Do not include explanations, reasoning, or extra words.
- Keep the answer in the **same language** as the question (Chinese → Chinese, English → English, Spanish → Spanish).
- If you are not confident you know the answer from your own knowledge, reply exactly: <NO_ANSWER>
Example outputs:
Q (English): What is the capital of France?
A: Paris
Q (Chinese): 中国的国庆节是哪一天？
A: 10月1日
Q (Spanish): ¿Cuál es el idioma oficial de España?
A: Español
Now answer the following question:
"""


def normalize_prediction(x):
    # IMPORTANT: handle NaN correctly
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if NO_ANSWER in s:
        return NO_ANSWER
    return s.split("\n")[0].strip()

def ask_ollama(question, system_prompt, model):
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()},
        ],
    )
    return resp["message"]["content"].strip()

def predict_one(question, model):
    a1 = ask_ollama(question, SYSTEM_STRICT, model)
    a1 = a1.split("\n")[0].strip()
    if NO_ANSWER in a1:
        a2 = ask_ollama(question, SYSTEM_FORCE, model)
        a2 = a2.split("\n")[0].strip()
        # guarantee no <NO_ANSWER> from force pass
        return a2 if NO_ANSWER not in a2 else a1
    return a1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pred_in", required=True)
    ap.add_argument("--pred_out", required=True)
    ap.add_argument("--redo_no_answer", action="store_true")
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    inp = pd.read_csv(args.input, sep="\t")
    pred = pd.read_csv(args.pred_in, sep="\t")

    # normalize only existing predictions, keep NA as NA
    pred["prediction"] = pred["prediction"].map(normalize_prediction)

    df = inp.merge(pred[["id", "prediction"]], on="id", how="left")

    # only en-
    is_en = df["id"].astype(str).str.startswith("en-")

    # decide what to predict
    if args.redo_no_answer:
        need = is_en & (df["prediction"].isna() | (df["prediction"] == NO_ANSWER))
    else:
        need = is_en & (df["prediction"].isna())

    idxs = df.index[need].tolist()
    if args.limit:
        idxs = idxs[:args.limit]

    print(f"Total input rows: {len(df)}")
    print(f"Need prediction (en- only): {len(idxs)} (redo_no_answer={args.redo_no_answer})")

    done = 0
    for i in idxs:
        q = str(df.at[i, "text"]).strip()
        if not q:
            df.at[i, "prediction"] = NO_ANSWER
            continue
        ans = predict_one(q, args.model)
        # final safety normalization
        df.at[i, "prediction"] = NO_ANSWER if NO_ANSWER in ans else ans
        done += 1
        if done % 50 == 0:
            print(f"...predicted {done}/{len(idxs)}")
        if args.sleep:
            time.sleep(args.sleep)

    # write
    out = df.loc[:, ["id", "prediction"]].copy()
    out["prediction"] = out["prediction"].apply(lambda x: NO_ANSWER if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x).split("\n")[0].strip())
    out.to_csv(args.pred_out, sep="\t", index=False)
    print(f"Wrote {args.pred_out}. Newly predicted: {done}")

if __name__ == "__main__":
    main()