# AlphaZero Rust Migration Plan

## Purpose

Speed up the long-running AlphaZero loop used by:

```bash
pnpm run alphazero:loop -- --iterations 20 --workers 8 --games 800 --teacher-ms 5000 --sample-size 100000 --epochs 8 --batch-size 2048 --evaluation-games 80 --candidate-ms 5000 --baseline-ms 5000 --max-buffer-shards 512 --max-buffer-samples 1000000 --replay-sample-strategy priority --start-policy fixedStart --min-elo-lower-bound-gain 0 --publish-best
```

The current path is Node.js self-play/MCTS, JSONL replay sampling, Python/PyTorch training, ONNX export, then Node.js arena evaluation. The expensive parts are self-play MCTS and repeated model inference. Training is already CUDA-capable through PyTorch, but the requested target is Rust with Burn for ML.

## Change Scope

- Add a Rust workspace under `crates/`.
- Port reusable Blokus Duo core logic to Rust first:
  - board/state representation
  - orientation/action encoding compatibility
  - legal move generation
  - state tensor encoding
  - scoring and terminal detection
- Add a Rust training CLI. The first compatibility backend can write JSONL, but the high-performance path should use a compact binary replay format.
- Add a Rust AlphaZero CLI that keeps the current report/replay/model-registry file layout.
- Move policy-value training to Burn in a later phase after core parity is proven.
- Keep browser runtime and GitHub Pages static build unchanged.

## Affected Files

- `Cargo.toml`
- `crates/blokus_core_rs/**`
- `crates/blokus_train_rs/**`
- `package.json`
- `training/run_alphazero_loop.mjs`
- `training/replay_buffer.mjs`
- `README.md`

## Implementation Steps

1. Rust core parity
   - Create `blokus_core_rs`.
   - Generate or embed piece orientations with the same global orientation ids as `packages/core/src/orientations.json`.
   - Add tests matching existing JS assertions:
     - 21 pieces
     - 91 orientations
     - `fixedStart` initial legal moves = 58
     - `chooseStart` initial legal moves = 116
     - action encode/decode compatibility
     - scoring compatibility

2. Fast self-play backend
   - Create `blokus_train_rs` CLI.
   - Output current JSONL schema:
     - `encoded_state`
     - `legal_actions`
     - `selected_action`
     - `policy_target_actions`
     - `policy_target_probs`
     - `final_score_diff`
   - Initially support a fast heuristic/MCTS engine without neural inference.
   - Wire Node `run_distributed_selfplay.mjs` to prefer Rust backend when available, with fallback to existing Node workers.

3. Neural inference for self-play
   - Add ONNX Runtime Rust or Burn model loading for policy-value inference.
   - Batch inference across MCTS leaves where possible.
   - Keep candidate ONNX output for browser compatibility unless replacing browser inference is explicitly requested.

4. Burn training
   - Add Burn policy-value model.
   - Train from compact binary replay format.
   - Use CUDA backend on Vast.
   - Export to ONNX or another browser-compatible format before `--publish-best` if the browser model should be updated.

Implemented Rust-native path:

- `blokus_burn_rs selfplay`: Rust PUCT self-play with Burn policy-value inference.
- `blokus_burn_rs train`: Burn trainer over binary replay using the Python-style conv/residual policy-value architecture and legal-action masked policy loss.
- `blokus_burn_rs merge`: merge worker binary replay shards.
- `blokus_burn_rs infer-smoke`: saved-model inference smoke.
- `alphazero:loop:rust`: accepts the existing AlphaZero loop option names and runs Rust binary self-play plus Burn training.

5. AlphaZero CLI integration
   - Add `alphazero:loop:rust` script.
   - Keep existing `alphazero:loop` stable until Rust loop passes parity and smoke tests.
   - Optionally switch `alphazero:loop` to Rust after validation.

## Verification

- `cargo test`
- `npm test`
- `npm run build`
- Small smoke:

```bash
pnpm run alphazero:loop:rust -- --iterations 1 --workers 2 --games 8 --teacher-ms 200 --sample-size 512 --epochs 1 --batch-size 128 --evaluation-games 4 --cpu
```

- Larger Vast GPU smoke:

```bash
pnpm run alphazero:loop:rust -- --iterations 1 --workers 8 --games 80 --teacher-ms 1000 --sample-size 10000 --epochs 1 --batch-size 2048 --evaluation-games 8
```

## Risks

- Burn CUDA crate versions and APIs may require adjustment against the installed Rust/CUDA environment.
- ONNX export from Burn may not be drop-in compatible with the current browser `onnxruntime-web` path.
- `alphazero:loop:rust` currently trains and publishes a Burn `.bin` model; arena gating and browser ONNX publishing still need a dedicated Rust/Burn evaluator/export path.
- A full MCTS + neural inference port changes training data distribution; arena gates should compare behavior before replacing the existing backend.
- The binary replay path intentionally drops JSONL compatibility for throughput.

## Rollback

- Keep current Node/Python commands untouched during migration.
- New Rust commands are additive until validated.
- If Rust training or inference fails on Vast, fall back to existing Node self-play and PyTorch training.
