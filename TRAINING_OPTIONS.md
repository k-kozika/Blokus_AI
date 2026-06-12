# Training Options

このファイルは、学習・dataset 生成・Critical Replay・arena 評価・self-play に関係する CLI オプションの一覧です。

共通ルール:

- npm script に引数を渡すときは `--` の後ろに書きます。
- GPU を使う場合は trainer の `--cpu` を付けません。
- CPU に固定したい場合だけ `--cpu` を付けます。
- Windows PowerShell で `npm.ps1` が止まる場合は `cmd /c npm ...` を使ってください。

例:

```bash
npm run train:policy-value -- --dataset training/data/smoke-expert-100.jsonl --epochs 3
```

## AI 名

多くの dataset / trajectory / arena コマンドで使えます。

```text
easy
normal
hard
expert
expert_plus
learned
master
```

`learned` には policy ONNX、`master` には policy-value ONNX を指定できます。

## Start Policy

主に以下を使います。

```text
fixedStart
chooseStart
```

通常の学習・評価は `fixedStart` が扱いやすいです。

## Python / GPU

### `npm run python:check`

Python、PyTorch、CUDA、ONNX の状態を表示します。

オプションはありません。

### `npm run python:install:gpu`

CUDA 13.0 対応 PyTorch と ONNX を入れます。

オプションはありません。

### `npm run python:install:cpu`

CPU 版 PyTorch と ONNX を入れます。

オプションはありません。

## Presets

### `npm run smoke:learning`

100 games の Expert dataset 生成、Policy / Policy-Value の 1 epoch 学習、ONNX export をまとめて実行します。

オプション:

| option | default | 説明 |
| --- | ---: | --- |
| `--cpu` | false | CPU に固定します。付けない場合は CUDA が使えれば GPU を使います。 |

### `npm run smoke:critical`

10 games trajectory、Master teacher の Critical Replay、Policy-Value 1 epoch 学習をまとめて実行します。

オプション:

| option | default | 説明 |
| --- | ---: | --- |
| `--cpu` | false | CPU に固定します。付けない場合は CUDA が使えれば GPU を使います。 |

## Dataset 生成

### `npm run generate:dataset`

AI 同士の対局から JSONL dataset を作ります。

| option | default | 説明 |
| --- | ---: | --- |
| `--games` | `100` | 生成するゲーム数。 |
| `--out` | `training/data/smoke-expert-100.jsonl` | 出力 JSONL path。 |
| `--teacher-ms` | `25` | 各手の AI 思考時間 ms。 |
| `--start-policy` | `fixedStart` | 初期配置方式。 |
| `--black-ai` | `expert` | 黒 AI。 |
| `--white-ai` | `expert` | 白 AI。 |
| `--black-model` | none | 黒が `learned` / `master` のときに使う ONNX。 |
| `--white-model` | none | 白が `learned` / `master` のときに使う ONNX。 |
| `--policy-target-source` | `auto` | `auto` / `visit` など。MCTS visit 分布を target にしたい場合は `visit`。 |
| `--parallel` | `1` | 複数 Node process で games を分割生成します。 |

例:

```bash
node training/generate_dataset.mjs --games 1000 --parallel 8 --out training/data/expert-1k.jsonl --black-ai expert --white-ai expert --teacher-ms 25
```

Learned self-play:

```bash
node training/generate_dataset.mjs --games 50000 --parallel 8 --out training/data/learned-selfplay-50k.jsonl --black-ai learned --white-ai learned --black-model apps/web/public/models/blokus_policy.onnx --white-model apps/web/public/models/blokus_policy.onnx
```

## Trajectory 生成

### `npm run generate:trajectory`

Critical Replay 用に、各手の state / legal actions / chosen action / 評価差分を保存します。

| option | default | 説明 |
| --- | ---: | --- |
| `--games` | `100` | 生成するゲーム数。 |
| `--out` | `training/dataset/trajectories/smoke` | 出力 directory。 |
| `--teacher-ms` | `50` | 各手の AI 思考時間 ms。 |
| `--start-policy` | `fixedStart` | 初期配置方式。 |
| `--ai` | none | 黒白の AI を同じ値にします。 |
| `--black-ai` | `expert` | 黒 AI。 |
| `--white-ai` | `expert` | 白 AI。 |
| `--black-model` | none | 黒が `learned` / `master` のときに使う ONNX。 |
| `--white-model` | none | 白が `learned` / `master` のときに使う ONNX。 |
| `--parallel` | `1` | 複数 Node process で games を分割生成します。 |

例:

```bash
node training/critical_state_replay/collect_trajectory.mjs --games 1000 --parallel 8 --ai learned --black-model apps/web/public/models/blokus_policy.onnx --white-model apps/web/public/models/blokus_policy.onnx --out training/dataset/trajectories/learned-1k
```

## Critical State Replay

### `npm run generate:critical-replay`

Trajectory から重要局面を選び、候補手ごとに teacher で分岐再生して `q_values` / `policy_probs` / `best_action` を作ります。

| option | default | 説明 |
| --- | ---: | --- |
| `--trajectories` | `training/dataset/trajectories/smoke` | trajectory directory または `.jsonl` path。 |
| `--out` | `training/dataset/critical_replay/smoke` | 出力 directory。 |
| `--critical-states-per-game` | `8` | 1 game から抽出する重要局面数。 |
| `--top-k-actions` | `8` | 各局面で評価する候補手数。 |
| `--playouts-per-action` | `2` | 候補手ごとの分岐再生数。 |
| `--teacher` | `expert` | 分岐再生に使う AI。`master` 推奨。 |
| `--teacher-model` | none | teacher が `master` / `learned` のときの ONNX。 |
| `--teacher-ms` | `25` | 分岐再生中の各手の思考時間 ms。 |
| `--temperature` | `0.7` | `q_values` から `policy_probs` を作る softmax 温度。低いほど best action に寄ります。 |
| `--epsilon` | `0.08` | 分岐再生中に上位候補からランダムに選ぶ確率。 |
| `--parallel` | `1` | trajectory を複数 Node process に分割して Critical Replay を生成します。 |

小規模確認:

```bash
node training/critical_state_replay/build_branch_dataset.mjs --trajectories training/dataset/trajectories/smoke-expert-10 --out training/dataset/critical_replay/smoke-expert-10 --critical-states-per-game 2 --top-k-actions 4 --playouts-per-action 1 --teacher master --teacher-model apps/web/public/models/blokus_policy_value.onnx --teacher-ms 300 --parallel 4
```

実運用寄り:

```bash
node training/critical_state_replay/build_branch_dataset.mjs --trajectories training/dataset/trajectories/learned-10k --out training/dataset/critical_replay/learned-master-10k --critical-states-per-game 4 --top-k-actions 5 --playouts-per-action 2 --teacher master --teacher-model apps/web/public/models/blokus_policy_value.onnx --teacher-ms 800 --parallel 8
```

## Policy 学習

### `npm run train:policy`

Policy-only model を学習します。Learned AI 用です。

| option | default | 説明 |
| --- | ---: | --- |
| `--dataset` | required | JSONL dataset path。 |
| `--output-dir` | `training/checkpoints/policy` | checkpoint / summary 出力先。 |
| `--epochs` | `1` | epoch 数。 |
| `--batch-size` | `2048` | batch size。 |
| `--lr` | `3e-4` | learning rate。 |
| `--weight-decay` | `1e-4` | AdamW weight decay。 |
| `--validation-split` | `0.1` | validation 比率。 |
| `--seed` | `7` | 乱数 seed。 |
| `--cpu` | false | CPU に固定。 |

例:

```bash
npm run train:policy -- --dataset training/data/expert-1k.jsonl --epochs 5 --batch-size 2048 --output-dir training/checkpoints/policy-expert-1k
```

## Policy-Value 学習

### `npm run train:policy-value`

Policy-Value model を学習します。Master AI 用です。通常 dataset、Critical Replay dataset、dataset mix config を読めます。

| option | default | 説明 |
| --- | ---: | --- |
| `--dataset` | required | JSONL dataset path、または JSON/YAML dataset mix config。 |
| `--output-dir` | `training/checkpoints/policy_value` | checkpoint / summary 出力先。 |
| `--epochs` | `1` | epoch 数。 |
| `--batch-size` | `2048` | batch size。 |
| `--lr` | `3e-4` | learning rate。 |
| `--weight-decay` | `1e-4` | AdamW weight decay。 |
| `--validation-split` | `0.1` | validation 比率。 |
| `--value-weight` | `0.5` | loss 内の value loss 重み。 |
| `--seed` | `7` | 乱数 seed。 |
| `--cpu` | false | CPU に固定。 |

例:

```bash
npm run train:policy-value -- --dataset training/configs/mixed-learned-critical.json --epochs 10 --batch-size 2048 --output-dir training/checkpoints/pv-learned-critical
```

## Dataset Mix Config

`train:policy-value -- --dataset <config.json>` で使います。

形式:

```json
{
  "datasets": [
    { "path": "../data/expert-10k.jsonl", "weight": 0.5 },
    { "path": "../data/learned-selfplay-50k.jsonl", "weight": 0.3 },
    { "path": "../dataset/critical_replay/learned-master-10k/records.jsonl", "weight": 0.2 }
  ]
}
```

注意:

- 相対 path は config file のある directory から解決されます。
- `weight` は sample weight として loss に反映されます。
- Critical Replay は最初は 0.2-0.3 程度が扱いやすいです。

## ONNX Export

### `npm run export:onnx`

Policy model を ONNX にします。

| option | default | 説明 |
| --- | ---: | --- |
| `--checkpoint` | required | `policy_latest.pt` など。 |
| `--out` | `apps/web/public/models/blokus_policy.onnx` | 出力 ONNX path。 |
| `--model-kind` | `policy` | `policy` / `policy_value`。通常この script では `policy`。 |

### `npm run export:onnx:pv`

Policy-Value model を ONNX にします。

| option | default | 説明 |
| --- | ---: | --- |
| `--checkpoint` | required | `policy_value_latest.pt` など。 |
| `--out` | `apps/web/public/models/blokus_policy.onnx` | 出力 ONNX path。通常は明示指定してください。 |
| `--model-kind` | `policy_value` | script 側で指定済み。 |

例:

```bash
npm run export:onnx:pv -- --checkpoint training/checkpoints/pv-learned-critical/policy_value_latest.pt --out training/reports/candidate-policy-value.onnx
```

## Arena 評価

### `npm run arena:ai`

2 つの AI を対戦評価します。

| option | default | 説明 |
| --- | ---: | --- |
| `--games` | `8` | 対局数。 |
| `--start-policy` | `fixedStart` | 初期配置方式。 |
| `--swap-colors` | `true` | `false` を指定すると色入れ替えなし。 |
| `--alpha-ai` | `learned` | alpha 側 AI。 |
| `--alpha-name` | `candidate` | summary 上の alpha 名。 |
| `--alpha-model` | none | alpha model path。 |
| `--alpha-ms` | `500` | alpha 思考時間 ms。 |
| `--beta-ai` | `expert` | beta 側 AI。 |
| `--beta-name` | `expert` | summary 上の beta 名。 |
| `--beta-model` | none | beta model path。 |
| `--beta-ms` | `300` | beta 思考時間 ms。 |
| `--out` | `training/reports/arena-summary.json` | summary 出力先。 |
| `--parallel` | `1` | 対局を複数 Node process に分割して並列実行します。 |

例:

```bash
node training/run_arena.mjs --games 100 --parallel 8 --alpha-ai master --alpha-model training/reports/candidate-policy-value.onnx --beta-ai expert_plus --beta-ms 300 --out training/reports/candidate-vs-expert-plus.json
```

CPU 使用率を上げたい場合は `--parallel` を増やします。`learned` / `master` は Node 側では ONNX Runtime Web の WASM 推論なので CUDA は使いません。並列 arena では各 process の ONNX Runtime thread 数を既定で `BLOKUS_ORT_THREADS=1` にしています。手動調整したい場合:

```bash
$env:BLOKUS_ORT_THREADS=2
node training/run_arena.mjs --games 100 --parallel 4 --alpha-ai learned --alpha-model apps/web/public/models/blokus_policy.onnx --beta-ai expert_plus --beta-ms 300 --out training/reports/learned-vs-expert-plus-100.json
```

## Learned Cycle

### `npm run cycle:learned`

Dataset 生成、Policy 学習、ONNX export、arena 評価をまとめて実行します。

| option | default | 説明 |
| --- | ---: | --- |
| `--games` | `100` | dataset 生成 games。 |
| `--start-policy` | `fixedStart` | 初期配置方式。 |
| `--black-ai` | `expert` | dataset 黒 AI。 |
| `--white-ai` | `expert` | dataset 白 AI。 |
| `--black-model` | none | 黒 model path。 |
| `--white-model` | none | 白 model path。 |
| `--teacher-ms` | `25` | dataset 生成時の思考時間 ms。 |
| `--epochs` | `1` | Policy 学習 epoch。 |
| `--batch-size` | `2048` | batch size。 |
| `--evaluation-games` | `8` | arena 評価 games。 |
| `--evaluation-opponent` | `expert` | 評価相手 AI。 |
| `--evaluation-opponent-ms` | `300` | 評価相手の思考時間 ms。 |
| `--candidate-ms` | `500` | candidate の思考時間 ms。 |
| `--dataset-out` | timestamp 配下 | dataset 出力先。 |
| `--checkpoint-dir` | timestamp 配下 | checkpoint 出力先。 |
| `--onnx-out` | timestamp 配下 | candidate ONNX 出力先。 |
| `--summary-out` | timestamp 配下 | summary 出力先。 |
| `--publish-model` | false | true なら browser default policy ONNX を上書き。 |
| `--cpu` | false | CPU に固定。 |

## Self-Play Loop

### `npm run selfplay:loop`

簡易 candidate-vs-best loop です。

| option | default | 説明 |
| --- | ---: | --- |
| `--iterations` | `1` | loop 回数。 |
| `--games` | `20` | iteration ごとの dataset games。 |
| `--teacher-ms` | `25` | dataset 生成時の思考時間 ms。 |
| `--epochs` | `1` | Policy-Value 学習 epoch。 |
| `--batch-size` | `2048` | batch size。 |
| `--cpu` | false | CPU に固定。 |
| `--evaluation-games` | `6` | candidate 評価 games。 |
| `--min-win-rate` | `0.55` | promote に必要な win rate。 |
| `--min-average-margin` | `0.5` | promote に必要な平均 score margin。 |
| `--publish-best` | false | true なら browser default policy-value ONNX を上書き。 |
| `--base-report-dir` | `training/reports/selfplay` | report root。 |
| `--best-model-path` | `training/models/best_policy_value.onnx` | best model path。 |

## Distributed Self-Play

### `npm run distributed:selfplay`

ローカル複数 worker で self-play shard を生成し、replay buffer に取り込みます。

| option | default | 説明 |
| --- | ---: | --- |
| `--workers` | `2` | worker 数。 |
| `--games` | `20` | 合計 games。 |
| `--teacher-ms` | `1000` | 各手の思考時間 ms。 |
| `--difficulty` | `master` | self-play AI。 |
| `--model-path` | none | model path。 |
| `--replay-buffer-dir` | `training/replay_buffer` | replay buffer directory。 |
| `--worker-output-dir` | `training/reports/distributed-selfplay` | worker 出力 directory。 |
| `--max-buffer-shards` | `64` | replay buffer shard 上限。 |
| `--max-buffer-samples` | `50000` | replay buffer sample 上限。 |
| `--start-policy` | `fixedStart` | 初期配置方式。 |
| `--policy-target-source` | `visit` | policy target source。 |
| `--shard-compression` | `gzip` | shard 圧縮。 |

## Multi-Host Coordinator / Agent

### `npm run coordinator:selfplay`

File-based queue に self-play job を積みます。

| option | default | 説明 |
| --- | ---: | --- |
| `--queue-dir` | `training/distributed_queue` | queue directory。 |
| `--output-dir` | `training/reports/distributed-queue` | job output directory。 |
| `--workers` | `2` | job 数。 |
| `--games` | `20` | 合計 games。 |
| `--teacher-ms` | `1000` | 各手の思考時間 ms。 |
| `--difficulty` | `master` | self-play AI。 |
| `--model-path` | none | model path。 |
| `--replay-buffer-dir` | `training/replay_buffer` | replay buffer directory。 |
| `--start-policy` | `fixedStart` | 初期配置方式。 |
| `--policy-target-source` | `visit` | policy target source。 |

### `npm run agent:selfplay`

Queue から job を claim して実行します。

| option | default | 説明 |
| --- | ---: | --- |
| `--queue-dir` | `training/distributed_queue` | queue directory。 |
| `--host-id` | `host-<pid>` | worker host id。 |
| `--max-jobs` | `1` | この agent が処理する最大 job 数。 |
| `--idle-ms` | `200` | pending job 待ちの sleep ms。 |

## AlphaZero Loop

### `npm run alphazero:loop`

Distributed self-play、replay buffer sampling、Policy-Value 学習、ONNX export、arena 評価、model registry 登録・promote 判定をまとめて実行します。

| option | default | 説明 |
| --- | ---: | --- |
| `--iterations` | `1` | loop 回数。 |
| `--workers` | `2` | distributed self-play worker 数。 |
| `--games` | `20` | iteration ごとの self-play games。 |
| `--teacher-ms` | `1000` | self-play 思考時間 ms。 |
| `--sample-size` | `4096` | replay buffer から学習 dataset に抽出する最大 sample 数。 |
| `--epochs` | `1` | Policy-Value 学習 epoch。 |
| `--batch-size` | `2048` | batch size。 |
| `--cpu` | false | CPU に固定。 |
| `--evaluation-games` | `6` | arena 評価 games。 |
| `--arena-parallel` | `1` | arena 評価を複数 Node process に分割して並列実行。 |
| `--evaluation-opponent` | active best / `expert` | 評価相手 AI。`expert_plus` / `expert` / `master` など。指定時は active best ではなくこの相手で gate します。 |
| `--evaluation-opponent-model` | none | 評価相手が `learned` / `master` のときに使う ONNX。 |
| `--candidate-ms` | `300` | candidate 思考時間 ms。 |
| `--baseline-ms` | `300` | baseline 思考時間 ms。 |
| `--min-win-rate` | `0.55` | 初回 baseline に対する promote 条件。 |
| `--min-average-margin` | `0.5` | 初回 baseline に対する promote 条件。 |
| `--min-elo-lower-bound-gain` | `0` | active best がある場合の Elo gate 条件。 |
| `--k-factor` | `24` | Elo 更新の K factor。 |
| `--publish-best` | false | promote 時に browser default model を上書き。 |
| `--replay-buffer-dir` | `training/replay_buffer` | replay buffer directory。 |
| `--registry-dir` | `training/model_registry` | model registry directory。 |
| `--base-report-dir` | `training/reports/alphazero` | report root。 |
| `--max-buffer-shards` | `64` | replay buffer shard 上限。 |
| `--max-buffer-samples` | `50000` | replay buffer sample 上限。 |
| `--start-policy` | `fixedStart` | 初期配置方式。 |
| `--replay-sample-strategy` | `priority` | replay sampling strategy。 |
| `--shard-compression` | `gzip` | shard 圧縮。 |

例: `expert_plus` を相手に並列評価する最小 loop。

```bash
npm run alphazero:loop -- --iterations 1 --workers 2 --games 20 --sample-size 1024 --epochs 1 --evaluation-games 8 --arena-parallel 4 --evaluation-opponent expert_plus --candidate-ms 300 --baseline-ms 300
```

例: 指定した model を `master` 相手として評価する場合。

```bash
npm run alphazero:loop -- --iterations 1 --workers 2 --games 20 --sample-size 1024 --epochs 1 --evaluation-games 8 --arena-parallel 4 --evaluation-opponent master --evaluation-opponent-model training/model_registry/models/model-0001.onnx
```

## Model Registry

### `npm run registry:model -- status`

Active model と model count を表示します。

| option | default | 説明 |
| --- | ---: | --- |
| `--registry-dir` | `training/model_registry` | registry directory。 |

### `npm run registry:model -- list`

Model 一覧を表示します。

| option | default | 説明 |
| --- | ---: | --- |
| `--registry-dir` | `training/model_registry` | registry directory。 |

### `npm run registry:model -- promote`

指定 model を手動 promote します。

| option | default | 説明 |
| --- | ---: | --- |
| `--registry-dir` | `training/model_registry` | registry directory。 |
| `--model-id` | required | promote する model id。 |
