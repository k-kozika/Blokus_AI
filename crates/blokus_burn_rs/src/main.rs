use blokus_core_rs::{
    ACTION_SIZE, BOARD_CELLS, BOARD_SIZE, Move, ORIENTATION_COUNT, STATE_PLANES, StartPolicy,
    apply_move, create_initial_state, encode_action, encode_state_tensor, generate_legal_moves,
    score_state,
};
use burn::module::{AutodiffModule, Module};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::Tensor;
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const INPUT_SIZE: usize = STATE_PLANES * BOARD_CELLS;
const MAGIC: &[u8; 8] = b"BLKSRP01";
const VALUE_HEAD_CHANNELS: usize = 32;
const POLICY_HEAD_CHANNELS: usize = 32;

#[cfg(feature = "cuda")]
type InferenceBackend = burn::backend::Cuda;
#[cfg(feature = "cuda")]
type TrainingBackend = burn::backend::Autodiff<burn::backend::Cuda>;

#[cfg(not(feature = "cuda"))]
type InferenceBackend = burn::backend::Flex;
#[cfg(not(feature = "cuda"))]
type TrainingBackend = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
}

impl<B: Backend> ResidualBlock<B> {
    fn new(device: &B::Device, channels: usize) -> Self {
        let conv = |bias| {
            Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .with_bias(bias)
                .init(device)
        };
        Self {
            conv1: conv(false),
            bn1: BatchNormConfig::new(channels).init(device),
            conv2: conv(false),
            bn2: BatchNormConfig::new(channels).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let out = relu(self.bn1.forward(self.conv1.forward(input)));
        relu(residual + self.bn2.forward(self.conv2.forward(out)))
    }
}

#[derive(Module, Debug)]
struct PolicyValueNet<B: Backend> {
    stem: Conv2d<B>,
    stem_bn: BatchNorm<B>,
    res1: ResidualBlock<B>,
    res2: ResidualBlock<B>,
    res3: ResidualBlock<B>,
    res4: ResidualBlock<B>,
    policy_conv1: Conv2d<B>,
    policy_bn1: BatchNorm<B>,
    policy_conv2: Conv2d<B>,
    pass_pool: AdaptiveAvgPool2d,
    pass_fc1: Linear<B>,
    pass_fc2: Linear<B>,
    value_conv: Conv2d<B>,
    value_bn: BatchNorm<B>,
    value_pool: AdaptiveAvgPool2d,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,
}

impl<B: Backend> PolicyValueNet<B> {
    fn new(device: &B::Device, channels: usize) -> Self {
        let channels = channels.max(8);
        Self {
            stem: Conv2dConfig::new([STATE_PLANES, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            stem_bn: BatchNormConfig::new(channels).init(device),
            res1: ResidualBlock::new(device, channels),
            res2: ResidualBlock::new(device, channels),
            res3: ResidualBlock::new(device, channels),
            res4: ResidualBlock::new(device, channels),
            policy_conv1: Conv2dConfig::new([channels, POLICY_HEAD_CHANNELS], [1, 1])
                .with_bias(false)
                .init(device),
            policy_bn1: BatchNormConfig::new(POLICY_HEAD_CHANNELS).init(device),
            policy_conv2: Conv2dConfig::new([POLICY_HEAD_CHANNELS, ORIENTATION_COUNT], [1, 1])
                .with_bias(true)
                .init(device),
            pass_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            pass_fc1: LinearConfig::new(channels, 32).init(device),
            pass_fc2: LinearConfig::new(32, 1).init(device),
            value_conv: Conv2dConfig::new([channels, VALUE_HEAD_CHANNELS], [1, 1])
                .with_bias(false)
                .init(device),
            value_bn: BatchNormConfig::new(VALUE_HEAD_CHANNELS).init(device),
            value_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            value_fc1: LinearConfig::new(VALUE_HEAD_CHANNELS, 64).init(device),
            value_fc2: LinearConfig::new(64, 1).init(device),
        }
    }

    fn trunk(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let features = relu(self.stem_bn.forward(self.stem.forward(input)));
        let features = self.res1.forward(features);
        let features = self.res2.forward(features);
        let features = self.res3.forward(features);
        self.res4.forward(features)
    }

    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let features = self.trunk(input);
        let policy_map = self.policy_conv2.forward(relu(
            self.policy_bn1
                .forward(self.policy_conv1.forward(features.clone())),
        ));
        let policy_logits = policy_map.flatten(1, 3);
        let pass_logits = self.pass_fc2.forward(relu(
            self.pass_fc1
                .forward(self.pass_pool.forward(features.clone()).flatten(1, 3)),
        ));
        let logits = Tensor::cat(vec![policy_logits, pass_logits], 1);
        let value = self
            .value_fc2
            .forward(relu(
                self.value_fc1.forward(
                    self.value_pool
                        .forward(relu(
                            self.value_bn.forward(self.value_conv.forward(features)),
                        ))
                        .flatten(1, 3),
                ),
            ))
            .tanh();
        (logits, value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplaySample {
    player: u8,
    state: Vec<f32>,
    legal_actions: Vec<u16>,
    policy_actions: Vec<u16>,
    policy_probs: Vec<f32>,
    value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplayFile {
    version: u32,
    samples: Vec<ReplaySample>,
}

#[derive(Debug, Clone)]
struct TrainConfig {
    replay: PathBuf,
    output: PathBuf,
    epochs: usize,
    batch_size: usize,
    channels: usize,
    lr: f64,
    value_weight: f64,
}

#[derive(Debug, Clone)]
struct SelfPlayConfig {
    games: usize,
    out: PathBuf,
    model: Option<PathBuf>,
    channels: usize,
    start_policy: StartPolicy,
    simulations: usize,
    time_ms: u64,
    candidate_limit: usize,
    exploration_c: f32,
    seed: u64,
}

#[derive(Debug, Clone)]
struct InferConfig {
    model: PathBuf,
    channels: usize,
}

#[derive(Debug, Clone)]
struct ExportWeightsConfig {
    model: PathBuf,
    out: PathBuf,
    channels: usize,
}

#[derive(Debug, Clone)]
struct MergeConfig {
    out: PathBuf,
    inputs: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
enum Command {
    Train(TrainConfig),
    SelfPlay(SelfPlayConfig),
    InferSmoke(InferConfig),
    ExportWeights(ExportWeightsConfig),
    Merge(MergeConfig),
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let swap = self.next_u32() as usize % (index + 1);
            values.swap(index, swap);
        }
    }
}

#[derive(Clone)]
struct EvalOutput {
    priors: Vec<f32>,
    value: f32,
}

struct Evaluator<B: Backend> {
    model: PolicyValueNet<B>,
    device: B::Device,
}

impl<B: Backend> Evaluator<B> {
    fn new(model: PolicyValueNet<B>, device: B::Device) -> Self {
        Self { model, device }
    }

    fn evaluate(&self, state: &blokus_core_rs::State) -> EvalOutput {
        let encoded = encode_state_tensor(state, state.current_player);
        let input = Tensor::<B, 1>::from_floats(encoded.as_slice(), &self.device).reshape([
            1,
            STATE_PLANES,
            BOARD_SIZE,
            BOARD_SIZE,
        ]);
        let (logits, value) = self.model.forward(input);
        let probs = softmax(logits, 1).into_data().to_vec::<f32>().unwrap();
        EvalOutput {
            priors: probs,
            value: value.into_data().to_vec::<f32>().unwrap()[0],
        }
    }
}

#[derive(Clone)]
struct SearchNode {
    state: blokus_core_rs::State,
    mv: Option<Move>,
    parent: Option<usize>,
    prior: f32,
    visits: usize,
    value_sum: f32,
    children: Vec<usize>,
    unexpanded: Vec<(Move, f32)>,
}

fn parse_args() -> Result<Command, String> {
    let args: Vec<String> = env::args().collect();
    let Some(command) = args.get(1).map(String::as_str) else {
        return Err(
            "Usage: blokus_burn_rs <train|selfplay|infer-smoke|export-weights|merge> ..."
                .to_owned(),
        );
    };

    match command {
        "train" => {
            let mut config = TrainConfig {
                replay: PathBuf::from("training/replay_buffer_rs/replay.bin"),
                output: PathBuf::from("training/checkpoints/burn-policy-value"),
                epochs: 1,
                batch_size: 2048,
                channels: 64,
                lr: 3e-4,
                value_weight: 0.5,
            };
            parse_options(&args[2..], |key, value| {
                match key {
                    "--replay" => config.replay = PathBuf::from(value),
                    "--output-dir" => config.output = PathBuf::from(value),
                    "--epochs" => config.epochs = value.parse().map_err(|_| "bad --epochs")?,
                    "--batch-size" => {
                        config.batch_size = value.parse().map_err(|_| "bad --batch-size")?
                    }
                    "--hidden" | "--channels" => {
                        config.channels = value.parse().map_err(|_| "bad --channels")?
                    }
                    "--lr" => config.lr = value.parse().map_err(|_| "bad --lr")?,
                    "--value-weight" => {
                        config.value_weight = value.parse().map_err(|_| "bad --value-weight")?
                    }
                    other => return Err(format!("Unknown train option: {other}")),
                };
                Ok(())
            })?;
            Ok(Command::Train(config))
        }
        "selfplay" => {
            let mut config = SelfPlayConfig {
                games: 100,
                out: PathBuf::from("training/replay_buffer_rs/replay.bin"),
                model: None,
                channels: 64,
                start_policy: StartPolicy::FixedStart,
                simulations: 128,
                time_ms: 0,
                candidate_limit: 96,
                exploration_c: 1.5,
                seed: 7,
            };
            parse_options(&args[2..], |key, value| {
                match key {
                    "--games" => config.games = value.parse().map_err(|_| "bad --games")?,
                    "--out" => config.out = PathBuf::from(value),
                    "--model" => config.model = Some(PathBuf::from(value)),
                    "--hidden" | "--channels" => {
                        config.channels = value.parse().map_err(|_| "bad --channels")?
                    }
                    "--start-policy" => {
                        config.start_policy = match value {
                            "chooseStart" => StartPolicy::ChooseStart,
                            "fixedStart" => StartPolicy::FixedStart,
                            _ => return Err("bad --start-policy".to_owned()),
                        }
                    }
                    "--simulations" => {
                        config.simulations = value.parse().map_err(|_| "bad --simulations")?
                    }
                    "--time-ms" => config.time_ms = value.parse().map_err(|_| "bad --time-ms")?,
                    "--candidate-limit" => {
                        config.candidate_limit =
                            value.parse().map_err(|_| "bad --candidate-limit")?
                    }
                    "--exploration-c" => {
                        config.exploration_c = value.parse().map_err(|_| "bad --exploration-c")?
                    }
                    "--seed" => config.seed = value.parse().map_err(|_| "bad --seed")?,
                    other => return Err(format!("Unknown selfplay option: {other}")),
                };
                Ok(())
            })?;
            Ok(Command::SelfPlay(config))
        }
        "infer-smoke" => {
            let mut config = InferConfig {
                model: PathBuf::from("training/checkpoints/burn-policy-value/model.bin"),
                channels: 64,
            };
            parse_options(&args[2..], |key, value| {
                match key {
                    "--model" => config.model = PathBuf::from(value),
                    "--hidden" | "--channels" => {
                        config.channels = value.parse().map_err(|_| "bad --channels")?
                    }
                    other => return Err(format!("Unknown infer-smoke option: {other}")),
                };
                Ok(())
            })?;
            Ok(Command::InferSmoke(config))
        }
        "export-weights" => {
            let mut config = ExportWeightsConfig {
                model: PathBuf::from("training/checkpoints/burn-policy-value/model.bin"),
                out: PathBuf::from("training/checkpoints/burn-policy-value/torch_weights.json"),
                channels: 64,
            };
            parse_options(&args[2..], |key, value| {
                match key {
                    "--model" => config.model = PathBuf::from(value),
                    "--out" => config.out = PathBuf::from(value),
                    "--hidden" | "--channels" => {
                        config.channels = value.parse().map_err(|_| "bad --channels")?
                    }
                    other => return Err(format!("Unknown export-weights option: {other}")),
                };
                Ok(())
            })?;
            Ok(Command::ExportWeights(config))
        }
        "merge" => {
            let mut out = PathBuf::from("training/replay_buffer_rs/replay.bin");
            let mut inputs = Vec::new();
            let mut index = 2;
            while index < args.len() {
                let value = args[index].as_str();
                if value == "--out" {
                    index += 1;
                    let Some(path) = args.get(index) else {
                        return Err("--out requires a value".to_owned());
                    };
                    out = PathBuf::from(path);
                } else {
                    inputs.push(PathBuf::from(value));
                }
                index += 1;
            }
            if inputs.is_empty() {
                return Err("merge requires at least one input replay".to_owned());
            }
            Ok(Command::Merge(MergeConfig { out, inputs }))
        }
        other => Err(format!("Unknown command: {other}")),
    }
}

fn parse_options<F>(args: &[String], mut f: F) -> Result<(), String>
where
    F: FnMut(&str, &str) -> Result<(), String>,
{
    let mut index = 0;
    while index < args.len() {
        let key = args[index].as_str();
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("{key} requires a value"))?;
        f(key, value)?;
        index += 1;
    }
    Ok(())
}

fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    Ok(())
}

fn write_replay(path: &Path, samples: Vec<ReplaySample>) -> Result<(), Box<dyn std::error::Error>> {
    ensure_parent(path)?;
    let replay = ReplayFile {
        version: 2,
        samples,
    };
    let bytes = bincode::serde::encode_to_vec(&replay, bincode::config::standard())?;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(MAGIC)?;
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

fn read_replay(path: &Path) -> Result<ReplayFile, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(format!("Not a blokus rust replay file: {}", path.display()).into());
    }
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    let (replay, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
    Ok(replay)
}

fn merge_replays(config: MergeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for input in &config.inputs {
        let replay = read_replay(input)?;
        samples.extend(replay.samples);
    }
    let sample_count = samples.len();
    write_replay(&config.out, samples)?;
    println!(
        "{}",
        serde_json::json!({
            "out": config.out,
            "inputs": config.inputs,
            "samples": sample_count,
        })
    );
    Ok(())
}

fn load_or_init_model<B: Backend>(
    path: Option<&Path>,
    hidden: usize,
    device: &B::Device,
) -> Result<PolicyValueNet<B>, Box<dyn std::error::Error>> {
    let mut model = PolicyValueNet::<B>::new(device, hidden);
    if let Some(path) = path {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        model = model.load_file(path, &recorder, device)?;
    }
    Ok(model)
}

fn save_model<B: Backend>(
    model: PolicyValueNet<B>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_parent(path)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    model.save_file(path, &recorder)?;
    Ok(())
}

fn tensor_values<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

fn tensor_shape<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<usize> {
    tensor.dims().to_vec()
}

fn add_tensor<B: Backend, const D: usize>(
    entries: &mut Vec<serde_json::Value>,
    name: &str,
    tensor: Tensor<B, D>,
) {
    let shape = tensor_shape(&tensor);
    let values = tensor_values(tensor);
    entries.push(serde_json::json!({
        "name": name,
        "shape": shape,
        "data": values,
    }));
}

fn add_linear_weight<B: Backend>(
    entries: &mut Vec<serde_json::Value>,
    name: &str,
    tensor: Tensor<B, 2>,
) {
    add_tensor(entries, name, tensor.transpose());
}

fn add_conv<B: Backend>(entries: &mut Vec<serde_json::Value>, prefix: &str, conv: &Conv2d<B>) {
    add_tensor(entries, &format!("{prefix}.weight"), conv.weight.val());
    if let Some(bias) = &conv.bias {
        add_tensor(entries, &format!("{prefix}.bias"), bias.val());
    }
}

fn add_linear<B: Backend>(entries: &mut Vec<serde_json::Value>, prefix: &str, linear: &Linear<B>) {
    add_linear_weight(entries, &format!("{prefix}.weight"), linear.weight.val());
    if let Some(bias) = &linear.bias {
        add_tensor(entries, &format!("{prefix}.bias"), bias.val());
    }
}

fn add_batch_norm<B: Backend>(
    entries: &mut Vec<serde_json::Value>,
    prefix: &str,
    bn: &BatchNorm<B>,
) {
    add_tensor(entries, &format!("{prefix}.weight"), bn.gamma.val());
    add_tensor(entries, &format!("{prefix}.bias"), bn.beta.val());
    add_tensor(
        entries,
        &format!("{prefix}.running_mean"),
        bn.running_mean.value(),
    );
    add_tensor(
        entries,
        &format!("{prefix}.running_var"),
        bn.running_var.value(),
    );
}

fn add_residual_block<B: Backend>(
    entries: &mut Vec<serde_json::Value>,
    prefix: &str,
    block: &ResidualBlock<B>,
) {
    add_conv(entries, &format!("{prefix}.block.0"), &block.conv1);
    add_batch_norm(entries, &format!("{prefix}.block.1"), &block.bn1);
    add_conv(entries, &format!("{prefix}.block.3"), &block.conv2);
    add_batch_norm(entries, &format!("{prefix}.block.4"), &block.bn2);
}

fn export_weights(config: ExportWeightsConfig) -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let model =
        load_or_init_model::<InferenceBackend>(Some(&config.model), config.channels, &device)?;
    let mut entries = Vec::new();
    add_conv(&mut entries, "trunk.stem.0", &model.stem);
    add_batch_norm(&mut entries, "trunk.stem.1", &model.stem_bn);
    add_residual_block(&mut entries, "trunk.trunk.0", &model.res1);
    add_residual_block(&mut entries, "trunk.trunk.1", &model.res2);
    add_residual_block(&mut entries, "trunk.trunk.2", &model.res3);
    add_residual_block(&mut entries, "trunk.trunk.3", &model.res4);
    add_conv(&mut entries, "policy.policy_head.0", &model.policy_conv1);
    add_batch_norm(&mut entries, "policy.policy_head.1", &model.policy_bn1);
    add_conv(&mut entries, "policy.policy_head.3", &model.policy_conv2);
    add_linear(&mut entries, "policy.pass_head.2", &model.pass_fc1);
    add_linear(&mut entries, "policy.pass_head.4", &model.pass_fc2);
    add_conv(&mut entries, "value.value_head.0", &model.value_conv);
    add_batch_norm(&mut entries, "value.value_head.1", &model.value_bn);
    add_linear(&mut entries, "value.value_head.5", &model.value_fc1);
    add_linear(&mut entries, "value.value_head.7", &model.value_fc2);
    ensure_parent(&config.out)?;
    let payload = serde_json::json!({
        "format": "blokus-burn-torch-state-dict-json",
        "model_kind": "policy_value",
        "channels": config.channels,
        "tensors": entries,
    });
    let mut writer = BufWriter::new(File::create(&config.out)?);
    serde_json::to_writer(&mut writer, &payload)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!(
        "{}",
        serde_json::json!({
            "out": config.out,
            "model": config.model,
            "tensors": entries.len(),
        })
    );
    Ok(())
}

fn dense_policy(sample: &ReplaySample) -> Vec<f32> {
    let mut target = vec![0.0; ACTION_SIZE];
    for (action, prob) in sample.policy_actions.iter().zip(sample.policy_probs.iter()) {
        target[*action as usize] = *prob;
    }
    target
}

fn dense_legal_mask(sample: &ReplaySample) -> Vec<f32> {
    let mut mask = vec![0.0; ACTION_SIZE];
    for action in &sample.legal_actions {
        mask[*action as usize] = 1.0;
    }
    mask
}

fn train(config: TrainConfig) -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let replay = read_replay(&config.replay)?;
    if replay.samples.is_empty() {
        return Err("Replay is empty".into());
    }

    let mut model = PolicyValueNet::<TrainingBackend>::new(&device, config.channels);
    let mut optim = AdamConfig::new().init::<TrainingBackend, PolicyValueNet<TrainingBackend>>();
    let mut indices: Vec<usize> = (0..replay.samples.len()).collect();
    let mut rng = Lcg::new(7);
    create_dir_all(&config.output)?;

    for epoch in 0..config.epochs {
        rng.shuffle(&mut indices);
        let mut losses = Vec::new();
        for chunk in indices.chunks(config.batch_size.max(1)) {
            let batch = chunk.len();
            let mut states = Vec::with_capacity(batch * INPUT_SIZE);
            let mut legal_masks = Vec::with_capacity(batch * ACTION_SIZE);
            let mut policies = Vec::with_capacity(batch * ACTION_SIZE);
            let mut values = Vec::with_capacity(batch);
            for index in chunk {
                let sample = &replay.samples[*index];
                states.extend_from_slice(&sample.state);
                legal_masks.extend_from_slice(&dense_legal_mask(sample));
                policies.extend_from_slice(&dense_policy(sample));
                values.push(sample.value);
            }
            let x = Tensor::<TrainingBackend, 1>::from_floats(states.as_slice(), &device)
                .reshape([batch, STATE_PLANES, BOARD_SIZE, BOARD_SIZE]);
            let legal_mask =
                Tensor::<TrainingBackend, 1>::from_floats(legal_masks.as_slice(), &device)
                    .reshape([batch, ACTION_SIZE]);
            let policy_target =
                Tensor::<TrainingBackend, 1>::from_floats(policies.as_slice(), &device)
                    .reshape([batch, ACTION_SIZE]);
            let value_target =
                Tensor::<TrainingBackend, 1>::from_floats(values.as_slice(), &device)
                    .reshape([batch, 1]);
            let (logits, predicted_value) = model.forward(x);
            let masked_logits = logits.mask_fill(legal_mask.lower_equal_elem(0.0), -1.0e9);
            let policy_loss = (policy_target * log_softmax(masked_logits, 1))
                .sum_dim(1)
                .mean()
                .neg();
            let value_loss = (predicted_value - value_target).square().mean();
            let loss = policy_loss + value_loss * config.value_weight;
            let loss_value = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(config.lr, model, grads);
            losses.push(loss_value);
        }
        let train_loss = losses.iter().sum::<f32>() / losses.len().max(1) as f32;
        println!(
            "{}",
            serde_json::json!({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "samples": replay.samples.len(),
                "backend": if cfg!(feature = "cuda") { "cuda" } else { "flex" },
            })
        );
        save_model(model.clone().valid(), &config.output.join("model.bin"))?;
    }
    Ok(())
}

fn legal_priors(
    output: &EvalOutput,
    legal_moves: &[Move],
    candidate_limit: usize,
    rng: &mut Lcg,
) -> Vec<(Move, f32)> {
    let mut entries: Vec<(Move, f32)> = legal_moves
        .iter()
        .map(|mv| (mv.clone(), output.priors[encode_action(mv)].max(1e-6)))
        .collect();
    rng.shuffle(&mut entries);
    entries.sort_by(|a, b| b.1.total_cmp(&a.1));
    entries.truncate(candidate_limit.max(1));
    let total = entries.iter().map(|entry| entry.1).sum::<f32>().max(1e-6);
    for entry in &mut entries {
        entry.1 /= total;
    }
    entries.reverse();
    entries
}

fn final_value(state: &blokus_core_rs::State, player: usize) -> f32 {
    let [black, white] = score_state(state);
    let diff = if player == 0 {
        black - white
    } else {
        white - black
    };
    (diff as f32 / 89.0).clamp(-1.0, 1.0)
}

fn puct_select(nodes: &[SearchNode], node_id: usize, exploration_c: f32) -> usize {
    let parent_visits = nodes[node_id].visits as f32 + 1.0;
    nodes[node_id]
        .children
        .iter()
        .copied()
        .max_by(|a, b| {
            let score = |id: usize| {
                let child = &nodes[id];
                let q = if child.visits == 0 {
                    0.0
                } else {
                    child.value_sum / child.visits as f32
                };
                q + exploration_c * child.prior * parent_visits.sqrt() / (1.0 + child.visits as f32)
            };
            score(*a).total_cmp(&score(*b))
        })
        .unwrap()
}

fn backpropagate(nodes: &mut [SearchNode], mut node_id: usize, value: f32, root_player: usize) {
    loop {
        let signed = if nodes[node_id].state.current_player == root_player {
            value
        } else {
            -value
        };
        nodes[node_id].visits += 1;
        nodes[node_id].value_sum += signed;
        if let Some(parent) = nodes[node_id].parent {
            node_id = parent;
        } else {
            break;
        }
    }
}

fn mcts_move<B: Backend>(
    root_state: &blokus_core_rs::State,
    evaluator: &Evaluator<B>,
    config: &SelfPlayConfig,
    rng: &mut Lcg,
) -> Result<(Move, Vec<u16>, Vec<f32>, f32), String> {
    let legal = generate_legal_moves(root_state);
    if legal.len() == 1 {
        let action = encode_action(&legal[0]) as u16;
        return Ok((legal[0].clone(), vec![action], vec![1.0], 0.0));
    }

    let root_player = root_state.current_player;
    let root_eval = evaluator.evaluate(root_state);
    let root_value = root_eval.value;
    let root_unexpanded = legal_priors(&root_eval, &legal, config.candidate_limit, rng);
    let mut nodes = vec![SearchNode {
        state: root_state.clone(),
        mv: None,
        parent: None,
        prior: 1.0,
        visits: 0,
        value_sum: 0.0,
        children: Vec::new(),
        unexpanded: root_unexpanded,
    }];
    let started = Instant::now();
    let deadline = (config.time_ms > 0).then(|| Duration::from_millis(config.time_ms));

    for _ in 0..config.simulations.max(1) {
        if deadline.is_some_and(|limit| started.elapsed() >= limit) {
            break;
        }
        let mut node_id = 0;
        while nodes[node_id].unexpanded.is_empty() && !nodes[node_id].children.is_empty() {
            node_id = puct_select(&nodes, node_id, config.exploration_c);
        }

        let value = if let Some((mv, prior)) = nodes[node_id].unexpanded.pop() {
            let child_state = apply_move(&nodes[node_id].state, &mv)?;
            let child_id = nodes.len();
            let eval = if child_state.status == blokus_core_rs::GameStatus::Finished {
                EvalOutput {
                    priors: Vec::new(),
                    value: final_value(&child_state, root_player),
                }
            } else {
                evaluator.evaluate(&child_state)
            };
            let child_legal = if child_state.status == blokus_core_rs::GameStatus::Playing {
                generate_legal_moves(&child_state)
            } else {
                Vec::new()
            };
            let unexpanded = if child_legal.is_empty() {
                Vec::new()
            } else {
                legal_priors(&eval, &child_legal, config.candidate_limit, rng)
            };
            nodes.push(SearchNode {
                state: child_state,
                mv: Some(mv),
                parent: Some(node_id),
                prior,
                visits: 0,
                value_sum: 0.0,
                children: Vec::new(),
                unexpanded,
            });
            nodes[node_id].children.push(child_id);
            node_id = child_id;
            eval.value
        } else if nodes[node_id].state.status == blokus_core_rs::GameStatus::Finished {
            final_value(&nodes[node_id].state, root_player)
        } else {
            evaluator.evaluate(&nodes[node_id].state).value
        };
        backpropagate(&mut nodes, node_id, value, root_player);
    }

    let children = nodes[0].children.clone();
    let total_visits = children
        .iter()
        .map(|id| nodes[*id].visits)
        .sum::<usize>()
        .max(1);
    let best = children
        .iter()
        .copied()
        .max_by_key(|id| nodes[*id].visits)
        .ok_or("MCTS did not expand root")?;
    let mut targets: Vec<(u16, f32, usize)> = children
        .iter()
        .map(|id| {
            let action = encode_action(nodes[*id].mv.as_ref().unwrap()) as u16;
            let visits = nodes[*id].visits;
            (action, visits as f32 / total_visits as f32, visits)
        })
        .collect();
    targets.sort_by(|a, b| b.2.cmp(&a.2));
    Ok((
        nodes[best].mv.clone().unwrap(),
        targets.iter().map(|entry| entry.0).collect(),
        targets.iter().map(|entry| entry.1).collect(),
        root_value,
    ))
}

fn selfplay(config: SelfPlayConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Lcg::new(config.seed);
    let device = Default::default();
    let model =
        load_or_init_model::<InferenceBackend>(config.model.as_deref(), config.channels, &device)?;
    let evaluator = Evaluator::new(model, device);
    let mut samples = Vec::new();

    for game_index in 0..config.games {
        let mut state = create_initial_state(config.start_policy);
        let mut game_samples = Vec::new();
        while state.status == blokus_core_rs::GameStatus::Playing {
            let player = state.current_player;
            let legal_moves = generate_legal_moves(&state);
            let (mv, policy_actions, policy_probs, root_value) =
                mcts_move(&state, &evaluator, &config, &mut rng)?;
            game_samples.push(ReplaySample {
                player: player as u8,
                state: encode_state_tensor(&state, player),
                legal_actions: legal_moves
                    .iter()
                    .map(|mv| encode_action(mv) as u16)
                    .collect(),
                policy_actions,
                policy_probs,
                value: root_value,
            });
            state = apply_move(&state, &mv)?;
        }
        let score = score_state(&state);
        for sample in &mut game_samples {
            let final_diff = if sample.player == 0 {
                score[0] - score[1]
            } else {
                score[1] - score[0]
            };
            sample.value = (final_diff as f32 / 89.0).clamp(-1.0, 1.0);
        }
        samples.extend(game_samples);
        eprintln!(
            "Rust PUCT game {}/{} score {}-{} total_samples={}",
            game_index + 1,
            config.games,
            score[0],
            score[1],
            samples.len()
        );
    }
    write_replay(&config.out, samples)?;
    println!(
        "{}",
        serde_json::json!({
            "out": config.out,
            "games": config.games,
            "backend": if cfg!(feature = "cuda") { "cuda" } else { "flex" },
            "simulations": config.simulations,
        })
    );
    Ok(())
}

fn infer_smoke(config: InferConfig) -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let model =
        load_or_init_model::<InferenceBackend>(Some(&config.model), config.channels, &device)?;
    let evaluator = Evaluator::new(model, device);
    let state = create_initial_state(StartPolicy::FixedStart);
    let output = evaluator.evaluate(&state);
    println!(
        "{}",
        serde_json::json!({
            "policy_len": output.priors.len(),
            "value": output.value,
            "backend": if cfg!(feature = "cuda") { "cuda" } else { "flex" },
        })
    );
    Ok(())
}

fn main() {
    let result = match parse_args() {
        Ok(Command::Train(config)) => train(config),
        Ok(Command::SelfPlay(config)) => selfplay(config),
        Ok(Command::InferSmoke(config)) => infer_smoke(config),
        Ok(Command::ExportWeights(config)) => export_weights(config),
        Ok(Command::Merge(config)) => merge_replays(config),
        Err(error) => Err(error.into()),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
