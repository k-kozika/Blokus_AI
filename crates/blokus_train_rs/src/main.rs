use blokus_core_rs::{
    Move, StartPolicy, apply_move, encode_action, encode_state_tensor, generate_legal_moves,
    piece_size, score_state,
};
use serde::Serialize;
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
struct Config {
    games: usize,
    out: PathBuf,
    start_policy: StartPolicy,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Sample {
    player: usize,
    actor_difficulty: String,
    encoded_state: Vec<f32>,
    legal_actions: Vec<usize>,
    selected_action: usize,
    expert_selected_action: usize,
    final_score_diff: isize,
    policy_target_actions: Vec<usize>,
    policy_target_probs: Vec<f32>,
    policy_target_visits: Vec<usize>,
    policy_target_total_visits: usize,
    root_value: Option<f32>,
    strategy: String,
}

#[derive(Debug, Clone, Serialize)]
struct Meta {
    games: usize,
    total_positions: usize,
    black_ai: String,
    white_ai: String,
    policy_target_source: String,
    start_policy: String,
    backend: String,
    elapsed_ms: u128,
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

    fn index(&mut self, len: usize) -> usize {
        if len <= 1 {
            0
        } else {
            self.next_u32() as usize % len
        }
    }
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        games: 100,
        out: PathBuf::from("training/data/rust-fast.jsonl"),
        start_policy: StartPolicy::FixedStart,
        seed: 7,
    };
    let args: Vec<String> = env::args().collect();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--games" => {
                index += 1;
                config.games = args
                    .get(index)
                    .ok_or("--games requires a value")?
                    .parse()
                    .map_err(|_| "--games must be an integer")?;
            }
            "--out" => {
                index += 1;
                config.out = PathBuf::from(args.get(index).ok_or("--out requires a value")?);
            }
            "--start-policy" => {
                index += 1;
                config.start_policy = match args.get(index).map(String::as_str) {
                    Some("chooseStart") => StartPolicy::ChooseStart,
                    Some("fixedStart") => StartPolicy::FixedStart,
                    Some(_) => {
                        return Err("--start-policy must be fixedStart or chooseStart".to_owned());
                    }
                    None => return Err("--start-policy requires a value".to_owned()),
                };
            }
            "--seed" => {
                index += 1;
                config.seed = args
                    .get(index)
                    .ok_or("--seed requires a value")?
                    .parse()
                    .map_err(|_| "--seed must be an integer")?;
            }
            "--teacher-ms" | "--difficulty" | "--model-path" | "--policy-target-source" => {
                index += 1;
            }
            unknown => return Err(format!("Unknown argument: {unknown}")),
        }
        index += 1;
    }
    Ok(config)
}

fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    Ok(())
}

fn move_piece_size(mv: &Move) -> usize {
    match mv {
        Move::Place { piece_id, .. } => piece_size(piece_id),
        Move::Pass { .. } => 0,
    }
}

fn choose_fast_move(moves: &[Move], rng: &mut Lcg) -> Move {
    let best_size = moves.iter().map(move_piece_size).max().unwrap_or(0);
    let candidates: Vec<&Move> = moves
        .iter()
        .filter(|mv| move_piece_size(mv) == best_size)
        .collect();
    (*candidates[rng.index(candidates.len())]).clone()
}

fn play_game(config: &Config, rng: &mut Lcg) -> Result<(Vec<Sample>, [isize; 2]), String> {
    let mut state = blokus_core_rs::create_initial_state(config.start_policy);
    let mut samples = Vec::new();

    while state.status == blokus_core_rs::GameStatus::Playing {
        let player = state.current_player;
        let moves = generate_legal_moves(&state);
        let legal_actions: Vec<usize> = moves.iter().map(encode_action).collect();
        let selected = choose_fast_move(&moves, rng);
        let selected_action = encode_action(&selected);
        samples.push(Sample {
            player,
            actor_difficulty: "rust_fast".to_owned(),
            encoded_state: encode_state_tensor(&state, player),
            legal_actions,
            selected_action,
            expert_selected_action: selected_action,
            final_score_diff: 0,
            policy_target_actions: vec![selected_action],
            policy_target_probs: vec![1.0],
            policy_target_visits: vec![1],
            policy_target_total_visits: 1,
            root_value: None,
            strategy: "rust_fast_largest_piece".to_owned(),
        });
        state = apply_move(&state, &selected)?;
    }

    let score = score_state(&state);
    for sample in &mut samples {
        sample.final_score_diff = if sample.player == 0 {
            score[0] - score[1]
        } else {
            score[1] - score[0]
        };
    }
    Ok((samples, score))
}

fn run(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    ensure_parent(&config.out)?;
    let mut writer = BufWriter::new(File::create(&config.out)?);
    let mut rng = Lcg::new(config.seed);
    let mut total_positions = 0usize;

    for game_index in 0..config.games {
        let (samples, score) = play_game(&config, &mut rng)?;
        total_positions += samples.len();
        for sample in samples {
            serde_json::to_writer(&mut writer, &sample)?;
            writer.write_all(b"\n")?;
        }
        eprintln!(
            "Rust generated game {}/{} (score {}-{})",
            game_index + 1,
            config.games,
            score[0],
            score[1]
        );
    }
    writer.flush()?;

    let meta = Meta {
        games: config.games,
        total_positions,
        black_ai: "rust_fast".to_owned(),
        white_ai: "rust_fast".to_owned(),
        policy_target_source: "selected".to_owned(),
        start_policy: match config.start_policy {
            StartPolicy::ChooseStart => "chooseStart".to_owned(),
            StartPolicy::FixedStart => "fixedStart".to_owned(),
        },
        backend: "blokus_train_rs".to_owned(),
        elapsed_ms: started.elapsed().as_millis(),
    };
    let mut meta_writer =
        BufWriter::new(File::create(format!("{}.meta.json", config.out.display()))?);
    serde_json::to_writer_pretty(&mut meta_writer, &meta)?;
    meta_writer.write_all(b"\n")?;
    println!("{}", serde_json::to_string_pretty(&meta)?);
    Ok(())
}

fn main() {
    match parse_args().and_then(|config| run(config).map_err(|error| error.to_string())) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
