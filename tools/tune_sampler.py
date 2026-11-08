#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_sampler.py - autonomous sampler search against the LIVE C++ engine over HTTP.

Why over HTTP and not a recompile per combo: the penalties and temperature now live in the C++
Sampler and are reachable per-request (repetition_penalty / presence_penalty / frequency_penalty /
temperature, see bonsai_http.h -> g_req -> fwd_run -> Sampler::pick). Sending them per request
exercises EXACTLY the same pick() code path a recompiled default would, so the search is valid
without paying a rebuild+reload per combo. The winning combo is then baked into the Sampler struct
defaults with ONE rebuild and re-verified with no overrides (see runs/overnight_tuning.log).

Fluency contract (the user's): >=100 characters of coherent answer, no "Wait" loop, no stutter
("这…个…是"), no degenerate n-gram repetition. Each combo is scored on 5 logic questions.

Reproducible: fixed seed per request, fixed question set, fixed ladder. Requires the daemon up:
  BONSAI_THINKING=0 BONSAI_NO_BROWSER=1 ./bonsai_vk.exe serve <model.gguf>
Run:  python tools/tune_sampler.py [--max-tokens 300] [--out runs/overnight_tuning.log]
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8080"

QUESTIONS = [
    "一个房间里有三个人，甲说乙在说谎，乙说丙在说谎，丙说甲和乙都在说谎。请问谁在说真话？请逐步推理。",
    "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain your reasoning step by step.",
    "有三扇门，一扇后面是汽车，两扇后面是羊。你选了一号门，主持人打开三号门后面是羊，问你是否换到二号门。换门是否提高中奖概率？为什么？",
    "一个数列 2, 6, 12, 20, 30, ... 请找出规律并给出第 10 项，说明推导过程。",
    "Why is the sky blue? Give a concise but complete physical explanation in at least three sentences.",
]

# (temperature, presence_penalty, repetition_penalty). The control row is the old greedy default.
LADDER = [
    ("control-greedy", 0.0, 0.0, 1.00),
    ("ladder-1", 0.7, 0.8, 1.10),
    ("ladder-2", 0.6, 0.6, 1.05),
    ("ladder-3", 0.6, 0.9, 1.10),
    ("ladder-4", 0.7, 1.0, 1.15),
    ("ladder-5", 0.8, 0.9, 1.10),
    ("ladder-6", 0.8, 1.1, 1.20),
    ("ladder-7", 0.9, 1.2, 1.20),
    ("ladder-8", 0.5, 0.5, 1.05),
]


def post(path, payload, timeout=240):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_status():
    try:
        with urllib.request.urlopen(BASE + "/v1/status", timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


def ask(params, question, max_tokens, seed=42):
    body = {
        "model": "bonsai",
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": params["temp"],
        "presence_penalty": params["presence"],
        "repetition_penalty": params["rep"],
    }
    t0 = time.time()
    j = post("/v1/chat/completions", body)
    wall = time.time() - t0
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = j.get("usage") or {}
    st = get_status()
    return {
        "content": content,
        "reasoning": reasoning,
        "finish": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "engine_tok_s": st.get("last_tok_s"),
        "wall_s": round(wall, 2),
    }


def metrics(text):
    """Heuristic fluency gate. Deliberately conservative and fully documented."""
    n = len(text)
    wait_hits = len(re.findall(r"\bwait\b", text, re.IGNORECASE))
    # stutter: a CJK char followed by an ellipsis/period, repeated >=3 times ("这…个…是")
    stutter = bool(re.search(r"(?:[一-鿿]\s*[….]{1,2}\s*){3,}", text))
    stutter = stutter or bool(re.search(r"(.)\1{4,}", text))
    # degenerate loop: an identical 6..40 char phrase repeated back-to-back >=3 times
    loop_phrase = bool(re.search(r"(.{6,40}?)\1{2,}", text, re.DOTALL))
    # degenerate loop: one 8-gram owns >25% of all 8-grams
    deg = False
    if n >= 24:
        grams = [text[i:i + 8] for i in range(n - 7)]
        counts = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        deg = (max(counts.values()) / float(len(grams))) > 0.25
    fluent = (n >= 100) and (wait_hits < 3) and (not stutter) and (not loop_phrase) and (not deg)
    return {"chars": n, "wait_hits": wait_hits, "stutter": stutter,
            "loop_phrase": loop_phrase, "degenerate": deg, "fluent": fluent}


def evaluate(label, temp, presence, rep, max_tokens, log):
    params = {"temp": temp, "presence": presence, "rep": rep}
    rows = []
    fluent_count = 0
    for qi, q in enumerate(QUESTIONS):
        try:
            r = ask(params, q, max_tokens)
        except Exception as e:  # a transport failure is a failed question, not a crash
            r = {"content": "", "reasoning": "", "finish": "error:" + str(e),
                 "completion_tokens": 0, "engine_tok_s": None, "wall_s": 0}
        m = metrics(r["content"])
        if m["fluent"]:
            fluent_count += 1
        rows.append({"q": qi, "metrics": m, "finish": r["finish"],
                     "tokens": r["completion_tokens"], "tok_s": r["engine_tok_s"],
                     "wall_s": r["wall_s"], "content": r["content"]})
    score = {"label": label, "temp": temp, "presence": presence, "rep": rep,
             "fluent": fluent_count, "of": len(QUESTIONS), "rows": rows}
    line = ("COMBO %-14s temp=%.2f pres=%.2f rep=%.2f  fluent=%d/%d" %
            (label, temp, presence, rep, fluent_count, len(QUESTIONS)))
    print(line, flush=True)
    log.write(line + "\n")
    for row in rows:
        m = row["metrics"]
        detail = ("  q%d chars=%d wait=%d stutter=%s loop=%s deg=%s fluent=%s finish=%s tok_s=%s" %
                  (row["q"], m["chars"], m["wait_hits"], m["stutter"], m["loop_phrase"],
                   m["degenerate"], m["fluent"], row["finish"], row["tok_s"]))
        log.write(detail + "\n")
        log.write("    text: " + row["content"][:220].replace("\n", " ") + "\n")
    log.flush()
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default="runs/overnight_tuning.log")
    ap.add_argument("--json", default="runs/tuning_search_v1.json")
    args = ap.parse_args()

    st = get_status()
    if not st:
        print("daemon not reachable at " + BASE, file=sys.stderr)
        return 2
    with open(args.out, "a", encoding="utf-8") as log:
        log.write("\n==== sampler search %s (thinking=off daemon, max_tokens=%d) ====\n"
                  % (time.strftime("%Y-%m-%d %H:%M:%S"), args.max_tokens))
        log.write("engine: model=%s cap=%s thinking=%s\n" %
                  (st.get("model"), st.get("cap"), st.get("thinking")))
        scores = []
        for label, t, p, r in LADDER:
            scores.append(evaluate(label, t, p, r, args.max_tokens, log))
        # Adaptive refinement around the best ladder row if it did not clear 4/5.
        best = max(scores, key=lambda s: (s["fluent"], sum(x["metrics"]["chars"] for x in s["rows"])))
        tries = 0
        while best["fluent"] < 4 and tries < 3:
            tries += 1
            loopish = sum(1 for s in scores for x in s["rows"]
                          if x["metrics"]["loop_phrase"] or x["metrics"]["degenerate"] or x["metrics"]["wait_hits"] >= 3)
            stutterish = sum(1 for s in scores for x in s["rows"] if x["metrics"]["stutter"])
            if loopish >= stutterish:
                nt = best["temp"]; np_ = min(1.2, best["presence"] + 0.15); nr = min(1.2, best["rep"] + 0.05)
            else:
                nt = best["temp"]; np_ = max(0.5, best["presence"] - 0.15); nr = max(1.05, best["rep"] - 0.05)
            cand = evaluate("refine-%d" % tries, nt, np_, nr, args.max_tokens, log)
            scores.append(cand)
            if cand["fluent"] > best["fluent"]:
                best = cand
        log.write("\nWINNER: label=%s temp=%.2f presence=%.2f rep=%.2f fluent=%d/%d\n" %
                  (best["label"], best["temp"], best["presence"], best["rep"], best["fluent"], best["of"]))
        log.flush()
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"scores": scores,
                   "winner": {k: best[k] for k in ("label", "temp", "presence", "rep", "fluent", "of")}},
                  f, ensure_ascii=False, indent=2)
    print("WINNER temp=%.2f presence=%.2f rep=%.2f fluent=%d/%d" %
          (best["temp"], best["presence"], best["rep"], best["fluent"], best["of"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())