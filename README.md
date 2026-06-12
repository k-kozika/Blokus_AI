# BlokusAI Duo

GitHub Pages で公開できる、2 人制 Blokus Duo 型の静的 AI 対戦アプリです。

対応モード:

- Human vs Human
- Human vs AI
- AI vs AI

実装済みの主な内容:

- 14 x 14 board
- start point `(0,0)` / `(13,13)`
- `chooseStart` / `fixedStart`
- 21 ピース、91 orientation
- 合法手生成
- 強制 pass
- 終局判定
- 公式スコア方式
- 探索型 AI difficulty 実装
- Learned Policy AI（Expert teacher imitation learning）
- Master Policy-Value MCTS AI
- ルール確認用の Human vs Human
- 棋譜 JSON copy / load
- GitHub Pages 用 static build
- AI / Worker / build の動作検証基盤

## Tech Stack

- Language: JavaScript ES modules
- Runtime: Node.js
- Package manager: npm
- Frontend: browser-native static app
- Build: custom Node.js copy build
- Testing: Node.js built-in assertions with suite runner
- Hosting: GitHub Pages
- CI/CD: GitHub Actions

ブラウザ推論には `onnxruntime-web` を使います。GitHub Pages で壊れにくいよう、ビルド後も相対パスで JS / CSS を読み込みます。

## Local Development

```bash
npm install
npm run dev
```

開発サーバーは既定で以下を使います。

```text
http://localhost:4173
```

ポートを変える場合:

```bash
$env:PORT=3000
npm run dev
```

## Build

```bash
npm run build
```

成果物は `dist/` に出力されます。

## Test

```bash
npm test
```

テストでは以下を確認します。

- ピース数が 21
- 各ピースの orientation 数
- orientation 合計が 91
- 各プレイヤーの総セル数が 89
- 初期合法手数 `chooseStart = 116`
- 初期合法手数 `fixedStart = 58`
- action encode / decode
- pass 制約
- scoring
- ランダム合法手の適用
- AI difficulty dispatch の正規化
- AI smoke test での合法手返却
- 手作り評価関数 `evaluateState` / `evaluateMoveQuick`
- Easy / Normal の Heuristic AI
- Hard の Beam Search AI
- Expert の Hybrid AI
- Expert+ の exact endgame solver
- Learned の ONNX policy inference
- Master の policy-value MCTS fallback
- Critical State Replay の branch dataset 生成
- TS / Python の action / state encoder 一致
- 終盤 Alpha-Beta Search と Transposition Table
- Worker protocol の `INIT` / `THINK` / `CANCEL`
- Worker fallback が違法手を安全な合法手へ置き換えること

## Verification

```bash
npm run verify
```

`npm run verify` は以下を順に実行します。

1. `npm test`
2. `npm run build`
3. `dist/` の build smoke check

## Other Checks

```bash
npm run lint
npm run typecheck
```

現在は外部 linter / TypeScript compiler を導入していないため、上記は確認用の no-op script です。

## Training Environment Quick Start

学習・GPU・Critical Replay の実行手順は `TRAINING_RUNBOOK.md` にまとめています。
学習関連 CLI の具体的な全オプションは `TRAINING_OPTIONS.md` にまとめています。

まず環境を確認する場合:

```bash
npm run python:check
```

CUDA 対応 PyTorch を入れる場合:

```bash
npm run python:install:gpu
```

最小の学習 smoke:

```bash
npm run smoke:learning
```

Critical Replay の最小 smoke:

```bash
npm run smoke:critical
```

学習コマンドのオプションを確認する場合:

```text
TRAINING_OPTIONS.md
```

## GitHub Pages Deploy

`.github/workflows/deploy.yml` が `main` branch への push で実行されます。

workflow の流れ:

1. repository を checkout
2. Node.js 22 をセットアップ
3. `npm ci`
4. `npm test`
5. `npm run build`
6. `dist/` を `gh-pages` branch に publish
7. GitHub Pages の Source を `Deploy from a branch` にして `gh-pages` / `root` を選ぶ

## Project Pages Base Path

このアプリは Vite などの bundler に依存せず、`./apps/...` と `./packages/...` の相対パスで静的ファイルを読み込みます。

そのため、以下のような Project Pages URL でも asset path が壊れない構成です。

```text
https://<user>.github.io/Blokus_AI/
```

User / Organization Pages や custom domain 直下でも同じ `dist/` を利用できます。

## Notes

- サーバー常駐処理、API Routes、SSR は使っていません。
- `.env` 系ファイルは `.gitignore` に含めています。
- AI は探索型 difficulty に加えて、Expert teacher を模倣した Learned policy を持ちます。
- `master` difficulty は policy-value model を使う MCTS + endgame search です。
- `npm test` と `npm run verify` でルール、AI legality、Worker protocol、build smoke を継続確認できます。

## Learned Policy Pipeline

共有 action / orientation 定義は以下で生成します。

```bash
npm run generate:orientations
```

100 局の smoke dataset 生成:

```bash
npm run generate:dataset -- --games 100 --out training/data/smoke-expert-100.jsonl --teacher-ms 10 --start-policy fixedStart
```

学習:

```bash
npm run train:policy -- --dataset training/data/smoke-expert-100.jsonl --epochs 1 --output-dir training/checkpoints/policy-100 --cpu
```

ONNX export:

```bash
npm run export:onnx -- --checkpoint training/checkpoints/policy-100/policy_latest.pt
```

出力モデル:

```text
apps/web/public/models/blokus_policy.onnx
```

`Learned` difficulty は browser 上で ONNX Runtime Web により推論し、モデル未ロード・推論失敗・違法手時は `Expert` に fallback します。

## Policy-Value Pipeline

Value Head 付きモデルの学習:

```bash
npm run train:policy-value -- --dataset training/data/smoke-expert-100.jsonl --epochs 1 --output-dir training/checkpoints/policy-value-100 --cpu
```

Policy-Value ONNX export:

```bash
npm run export:onnx:pv -- --checkpoint training/checkpoints/policy-value-100/policy_value_latest.pt --out apps/web/public/models/blokus_policy_value.onnx
```

出力モデル:

```text
apps/web/public/models/blokus_policy_value.onnx
```

`Master` difficulty は browser 上で policy-value model を読み、policy prior と value を使う MCTS を行います。推論失敗時は `Expert` に fallback します。

## AI vs AI Training Environment

AI 同士の対戦、dataset 生成、学習、ONNX export、候補モデル評価をコマンドで回せます。

### 1. 任意 AI の対戦 dataset を生成

`expert` 同士だけでなく、`easy / normal / hard / expert / expert_plus / learned` の任意組み合わせから教師データを生成できます。

```bash
npm run generate:dataset -- --games 100 --out training/data/arena-100.jsonl --teacher-ms 25 --black-ai expert --white-ai learned --white-model apps/web/public/models/blokus_policy.onnx
```

出力サンプルには以下が入ります。

- `encoded_state`
- `legal_actions`
- `selected_action`
- `expert_selected_action`（既存互換）
- `final_score_diff`

### 2. Node 上で AI vs AI アリーナを回す

```bash
npm run arena:ai -- --games 8 --alpha-ai learned --alpha-model apps/web/public/models/blokus_policy.onnx --beta-ai expert --beta-ms 300 --out training/reports/arena-summary.json
```

既定では `swapColors=true` なので、奇数ゲームごとに黒白を入れ替えて比較します。

### 3. 学習サイクルをまとめて回す

以下は、

1. dataset 生成
2. Policy 学習
3. ONNX export
4. candidate vs opponent のアリーナ評価

をまとめて実行します。

```bash
npm run cycle:learned -- --games 100 --epochs 1 --cpu --evaluation-games 8 --evaluation-opponent expert
```

主なオプション:

- `--black-ai`, `--white-ai`: dataset 生成時の対戦 AI
- `--black-model`, `--white-model`: Learned 側に使う ONNX
- `--checkpoint-dir`: 学習 checkpoint 出力先
- `--onnx-out`: 候補 ONNX 出力先
- `--publish-model true`: 候補 ONNX を `apps/web/public/models/blokus_policy.onnx` に反映

出力:

- dataset: `training/reports/<timestamp>/dataset.jsonl`
- checkpoint: `training/reports/<timestamp>/checkpoints/`
- candidate ONNX: `training/reports/<timestamp>/candidate.onnx`
- summary: `training/reports/<timestamp>/cycle-summary.json`

### 4. 典型的な流れ

100 局で smoke:

```bash
npm run cycle:learned -- --games 100 --epochs 1 --cpu --evaluation-games 4
```

1000 局で比較:

```bash
npm run cycle:learned -- --games 1000 --epochs 5 --cpu --evaluation-games 10 --evaluation-opponent expert
```

`publish-model` を付けない限り、ブラウザの既定モデルは上書きしません。

## Self-Play / Best Model Loop

最小の candidate-vs-best 更新ループを回すには以下を使います。

```bash
npm run selfplay:loop -- --iterations 1 --games 20 --epochs 1 --cpu --evaluation-games 6
```

このコマンドは以下を行います。

1. best model もしくは Expert を使って dataset 生成
2. Policy-Value model 学習
3. candidate ONNX export
4. candidate vs best の AI vs AI 評価
5. 勝率条件を満たした場合のみ best model 更新

既定の best model 保存先:

```text
training/models/best_policy_value.onnx
```

## Replay Buffer

file-backed replay buffer を使って self-play shard を蓄積できます。

- manifest: `training/replay_buffer*/manifest.json`
- shard: `training/replay_buffer*/shards/*.jsonl.gz`

buffer には以下が入ります。

- `selected_action`
- `policy_target_actions`
- `policy_target_probs`
- `final_score_diff`

`master` self-play の場合、MCTS root の visit 分布を疎な policy target として保存します。

AlphaZero loop では既定で `priority` sampling を使います。sample 側の `sampling_priority` と shard 側の `priority` を組み合わせて、uniform より重要局面を引きやすくしています。

## Critical State Replay

教師 AI の対局軌跡から重要局面を抽出し、その局面から候補手ごとに分岐再生して policy-value 学習用の target を作れます。

対局軌跡の保存:

```bash
npm run generate:trajectory -- --games 100 --ai expert --teacher-ms 50 --out training/dataset/trajectories/expert-100
```

重要局面の分岐再生 dataset 生成:

```bash
npm run generate:critical-replay -- --trajectories training/dataset/trajectories/expert-100 --out training/dataset/critical_replay/expert-100 --critical-states-per-game 8 --top-k-actions 8 --playouts-per-action 2
```

生成される `records.jsonl` には以下が入ります。

- `encoded_state`
- `legal_actions`
- `evaluated_actions`
- `q_values`
- `policy_probs`
- `best_action`
- `blunder_score`
- `sampling_priority`

Policy-Value 学習器は通常 dataset と Critical State Replay dataset の両方を読めます。

```bash
npm run train:policy-value -- --dataset training/dataset/critical_replay/expert-100/records.jsonl --epochs 1 --output-dir training/checkpoints/critical-replay-100 --cpu
```

複数 dataset を混ぜる場合は JSON config を指定できます。

```json
{
  "datasets": [
    { "path": "../data/selfplay.jsonl", "weight": 1.0 },
    { "path": "../critical_replay/expert-100/records.jsonl", "weight": 1.0 }
  ]
}
```

## Distributed Self-Play

複数 worker process を使って self-play shard を並列生成できます。

```bash
npm run distributed:selfplay -- --workers 4 --games 40 --teacher-ms 300 --difficulty master --model-path apps/web/public/models/blokus_policy_value.onnx
```

主なオプション:

- `--workers`: 並列 worker 数
- `--games`: 合計 self-play game 数
- `--replay-buffer-dir`: 取り込み先 replay buffer
- `--worker-output-dir`: worker の一時 shard 出力先
- `--max-buffer-shards`: 保持 shard 上限
- `--max-buffer-samples`: 保持 sample 上限

この方式は shared filesystem を前提にしたローカル distributed self-play 基盤です。

## Multi-Host Distributed Self-Play

複数ホストから同じ shared directory を見られる場合は、file-based queue で job を分配できます。

coordinator:

```bash
npm run coordinator:selfplay -- --queue-dir training/distributed_queue --output-dir training/reports/distributed-queue --workers 8 --games 80 --teacher-ms 300 --difficulty master --replay-buffer-dir training/replay_buffer
```

agent:

```bash
npm run agent:selfplay -- --queue-dir training/distributed_queue --host-id host-a --max-jobs 4
```

coordinator が pending job を積み、各 host の agent が claim して self-play shard を生成し、replay buffer に gzip で取り込みます。

## Model Registry

best model の登録と昇格は registry で管理します。

既定の保存先:

```text
training/model_registry/
```

状態確認:

```bash
npm run registry:model -- status
```

一覧表示:

```bash
npm run registry:model -- list
```

手動 promote:

```bash
npm run registry:model -- promote --model-id model-0001
```

registry には以下を保持します。

- model metadata
- promotion history
- rating
- ratingGames
- ratingHistory

## AlphaZero Loop

policy-value self-play、replay buffer sampling、学習、candidate export、arena 評価、best model promotion をまとめて回すには以下を使います。

```bash
npm run alphazero:loop -- --iterations 1 --workers 4 --games 40 --sample-size 4096 --epochs 2 --evaluation-games 8 --cpu
```

流れ:

1. active best model または baseline model で distributed self-play
2. replay buffer から training dataset を sample
3. `train_policy_value.py` で学習
4. candidate policy-value ONNX を export
5. candidate vs active best / baseline を arena で比較
6. Elo lower bound を含む gate を超えたら registry の active best を更新

出力:

- reports: `training/reports/alphazero/`
- replay buffer: `training/replay_buffer/`
- model registry: `training/model_registry/`

## Current Scope

いまの実装でできること:

- imitation learning の Learned AI
- policy-value model を使う Master AI
- replay buffer つき self-play
- distributed self-play worker / coordinator
- best model registry
- candidate-vs-best AlphaZero-style loop

まだ将来拡張のもの:

- multi-host を越えた network coordinator / service 化
- replay buffer の優先 sampling 戦略の高度化
- Elo gating の長期安定化と対戦相手プール拡張
- GPU 学習を含む CI
