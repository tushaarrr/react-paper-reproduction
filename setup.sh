#!/usr/bin/env bash
set -e
echo "==> python env"
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> paper"
[ -f paper/paper.pdf ] || curl -L -o paper/paper.pdf https://arxiv.org/pdf/2210.03629

echo "==> reference repo (read-only: notebooks, wikienv.py, wrappers.py)"
[ -d reference ] || git clone -q --depth 1 https://github.com/ysymyth/ReAct.git reference

# data/ is gitignored (CLAUDE.md rule 9); restore it from the reference clone.
mkdir -p data prompts
cp -n reference/data/*.json reference/data/*.jsonl data/ 2>/dev/null || true
cp -n reference/prompts/prompts_naive.json reference/prompts/fever.json prompts/ 2>/dev/null || true

echo "==> check"
python3 - <<'PY'
import json, random
d = json.load(open('data/hotpot_dev_v1_simplified.json'))
idx = list(range(7405)); random.Random(233).shuffle(idx)
assert len(d) == 7405, len(d)
assert idx[:5] == [3687, 6238, 5388, 3522, 3824], idx[:5]
p = json.load(open('prompts/prompts_naive.json'))
assert 'webthink_simple6' in p and 'webact_simple6' in p
print("dev questions:", len(d))
print("first 5 eval indices:", idx[:5], "OK")
PY

echo
echo "Done. Next:"
echo "  cp .env.example .env   # add MODEL and your API key"
echo "  claude                 # then paste prompts/claude-code/01-summarize.md"
