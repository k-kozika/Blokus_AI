# BlokusAI Duo Training Runbook

このメモは、人がローカルで学習・検証を回すための最短手順です。

学習関連 CLI の全オプションは `TRAINING_OPTIONS.md` にあります。この runbook は「何をどの順で実行するか」、`TRAINING_OPTIONS.md` は「各コマンドのオプション名、既定値、意味」を確認するためのものです。

## 1. 初回セットアップ

Windows の PowerShell で `npm.ps1` が止まる場合は、`cmd` または `cmd /c` で実行してください。

```bash
npm install
npm run python:install:gpu
npm run python:check
```

CPU だけで動かす場合:

```bash
npm run python:install:cpu
npm run python:check
```

`python:check` で確認するポイント:

- `torch` が表示される
- GPU 学習したい場合は `cuda_available: true`
- `onnx` が表示される

## 2. 基本検証

```bash
npm test
npm run verify
npm run build
```

## 3. 最小の学習 smoke

GPU が使える場合:

```bash
npm run smoke:learning
```

CPU で回す場合:

```bash
npm run smoke:learning -- --cpu
```

この preset は以下を実行します。

1. orientation 生成
2. Expert 100 games dataset 生成
3. Policy 1 epoch 学習
4. Policy-Value 1 epoch 学習
5. Learned / Master 用 ONNX export

## 4. Critical Replay smoke

GPU が使える場合:

```bash
npm run smoke:critical
```

CPU で回す場合:

```bash
npm run smoke:critical -- --cpu
```

この preset は以下を実行します。

1. Expert 10 games trajectory 生成
2. Master teacher で Critical State Replay 生成
3. Critical Replay records だけで Policy-Value 1 epoch 学習

## 5. 推奨する役割分担

- Learned: 高速 self-play で局面分布を増やす
- Expert: 安定した教師・fallback として使う
- Master: 大量 self-play の主役ではなく、重要局面の再評価、Critical Replay、candidate 評価に使う

Master は高品質ですが重いので、全局面で大量生成に使うより、重要局面だけを深く評価する用途に寄せます。

## 6. 実運用の dataset mix 例

```bash
npm run train:policy-value -- --dataset training/configs/mixed-learned-critical.json --epochs 10 --output-dir training/checkpoints/pv-learned-critical
```

GPU を強制的に避けたい場合だけ `--cpu` を追加します。

```bash
npm run train:policy-value -- --dataset training/configs/mixed-learned-critical.json --epochs 10 --output-dir training/checkpoints/pv-learned-critical --cpu
```

## 7. 評価

```bash
npm run export:onnx:pv -- --checkpoint training/checkpoints/pv-learned-critical/policy_value_latest.pt --out training/reports/candidate-policy-value.onnx

npm run arena:ai -- --games 100 --alpha-ai master --alpha-model training/reports/candidate-policy-value.onnx --beta-ai expert --beta-ms 300 --out training/reports/candidate-master-vs-expert.json

npm run arena:ai -- --games 100 --alpha-ai master --alpha-model training/reports/candidate-policy-value.onnx --beta-ai expert_plus --beta-ms 300 --out training/reports/candidate-master-vs-expert-plus.json
```

見る指標:

- wins / losses / draws
- averageMargin
- illegal move が発生していないこと
- fallback / timeout が増えていないこと

## 8. 注意

- `BLOKUS_PYTHON` を指定すると、その Python を優先します。
- 指定しない場合は `.venv`、通常の `python`、最後に Codex bundled Python の順で探します。
- Critical Replay の比率は最初は 20-30% 程度に抑えるのが扱いやすいです。
- Learned self-play だけで再学習を回し続けると弱点を増幅しやすいので、Expert と Master Critical Replay を混ぜてください。

## 9. オプション早見表

詳細は `TRAINING_OPTIONS.md` を見てください。

よく使う dataset 生成オプション:

```bash
npm run generate:dataset -- \
  --games 1000 \
  --parallel 8 \
  --out training/data/expert-1k.jsonl \
  --teacher-ms 25 \
  --start-policy fixedStart \
  --black-ai expert \
  --white-ai expert
```

Learned / Master を使う場合は model path を付けます。

```bash
npm run generate:dataset -- \
  --games 1000 \
  --parallel 8 \
  --out training/data/learned-selfplay-1k.jsonl \
  --black-ai learned \
  --white-ai learned \
  --black-model apps/web/public/models/blokus_policy.onnx \
  --white-model apps/web/public/models/blokus_policy.onnx
```

よく使う Critical Replay オプション:

```bash
npm run generate:critical-replay -- \
  --trajectories training/dataset/trajectories/learned-1k \
  --out training/dataset/critical_replay/learned-master-1k \
  --critical-states-per-game 4 \
  --top-k-actions 5 \
  --playouts-per-action 2 \
  --parallel 8 \
  --teacher master \
  --teacher-model apps/web/public/models/blokus_policy_value.onnx \
  --teacher-ms 800 \
  --temperature 0.7 \
  --epsilon 0.08
```

よく使う Policy-Value 学習オプション:

```bash
npm run train:policy-value -- \
  --dataset training/configs/mixed-learned-critical.json \
  --epochs 10 \
  --batch-size 2048 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --validation-split 0.1 \
  --value-weight 0.5 \
  --output-dir training/checkpoints/pv-learned-critical
```

CPU に固定する場合だけ `--cpu` を追加します。

```bash
npm run train:policy-value -- \
  --dataset training/configs/mixed-learned-critical.json \
  --epochs 10 \
  --output-dir training/checkpoints/pv-learned-critical-cpu \
  --cpu
```

評価でよく使うオプション:

```bash
npm run arena:ai -- \
  --games 100 \
  --start-policy fixedStart \
  --swap-colors true \
  --alpha-ai master \
  --alpha-model training/reports/candidate-policy-value.onnx \
  --alpha-ms 500 \
  --beta-ai expert_plus \
  --beta-ms 300 \
  --out training/reports/candidate-vs-expert-plus.json
```
