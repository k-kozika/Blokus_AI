use serde::Deserialize;
use std::collections::HashSet;
use std::sync::OnceLock;

pub const BOARD_SIZE: usize = 14;
pub const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE;
pub const EMPTY: i8 = -1;
pub const STATE_PLANES: usize = 51;
pub const ORIENTATION_COUNT: usize = 91;
pub const PASS_ACTION: usize = ORIENTATION_COUNT * BOARD_CELLS;
pub const ACTION_SIZE: usize = PASS_ACTION + 1;

pub const PIECE_IDS: [&str; 21] = [
    "I1", "I2", "I3", "V3", "I4", "O4", "T4", "L4", "Z4", "F5", "I5", "L5", "P5", "T5", "U5", "V5",
    "W5", "X5", "Y5", "Z5", "N5",
];

const ORTHOGONAL_DIRS: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONAL_DIRS: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

#[derive(Debug, Clone, Deserialize)]
pub struct Orientation {
    #[serde(rename = "pieceId")]
    pub piece_id: String,
    #[serde(rename = "localId")]
    pub local_id: usize,
    #[serde(rename = "globalId")]
    pub global_id: usize,
    pub cells: Vec<[isize; 2]>,
    pub width: usize,
    pub height: usize,
    #[serde(rename = "unitSquares")]
    pub unit_squares: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartPolicy {
    ChooseStart,
    FixedStart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartPoint {
    A,
    B,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Move {
    Place {
        player: usize,
        piece_id: String,
        orientation_global_id: usize,
        x: usize,
        y: usize,
    },
    Pass {
        player: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameStatus {
    Playing,
    Finished,
}

#[derive(Debug, Clone)]
pub struct State {
    pub board: [i8; BOARD_CELLS],
    pub current_player: usize,
    pub turn: usize,
    pub status: GameStatus,
    pub start_policy: StartPolicy,
    pub start_assignment: [Option<StartPoint>; 2],
    pub remaining_pieces: [Vec<String>; 2],
    pub placed_pieces: [Vec<String>; 2],
    pub last_placed_piece: [Option<String>; 2],
    pub forced_passed: [bool; 2],
    pub consecutive_passes: usize,
}

static ORIENTATIONS: OnceLock<Vec<Orientation>> = OnceLock::new();

pub fn orientations() -> &'static [Orientation] {
    ORIENTATIONS
        .get_or_init(|| {
            serde_json::from_str(include_str!("../../../packages/core/src/orientations.json"))
                .expect("packages/core/src/orientations.json must be valid")
        })
        .as_slice()
}

pub fn orientation_count() -> usize {
    orientations().len()
}

pub fn action_size() -> usize {
    ACTION_SIZE
}

pub fn pass_action() -> usize {
    PASS_ACTION
}

pub fn create_initial_state(start_policy: StartPolicy) -> State {
    let fixed = start_policy == StartPolicy::FixedStart;
    State {
        board: [EMPTY; BOARD_CELLS],
        current_player: 0,
        turn: 0,
        status: GameStatus::Playing,
        start_policy,
        start_assignment: if fixed {
            [Some(StartPoint::A), Some(StartPoint::B)]
        } else {
            [None, None]
        },
        remaining_pieces: [piece_ids_vec(), piece_ids_vec()],
        placed_pieces: [Vec::new(), Vec::new()],
        last_placed_piece: [None, None],
        forced_passed: [false, false],
        consecutive_passes: 0,
    }
}

fn piece_ids_vec() -> Vec<String> {
    PIECE_IDS.iter().map(|piece| (*piece).to_owned()).collect()
}

pub fn other(player: usize) -> usize {
    if player == 0 { 1 } else { 0 }
}

fn cell_index(x: usize, y: usize) -> usize {
    y * BOARD_SIZE + x
}

fn in_bounds(x: isize, y: isize) -> bool {
    x >= 0 && y >= 0 && x < BOARD_SIZE as isize && y < BOARD_SIZE as isize
}

fn get_cell(board: &[i8; BOARD_CELLS], x: isize, y: isize) -> i8 {
    if !in_bounds(x, y) {
        return EMPTY;
    }
    board[cell_index(x as usize, y as usize)]
}

fn set_cell(board: &mut [i8; BOARD_CELLS], x: usize, y: usize, player: usize) {
    board[cell_index(x, y)] = player as i8;
}

pub fn get_orientation(global_id: usize) -> Option<&'static Orientation> {
    orientations().get(global_id)
}

pub fn get_orientations(piece_id: &str) -> Vec<&'static Orientation> {
    orientations()
        .iter()
        .filter(|orientation| orientation.piece_id == piece_id)
        .collect()
}

pub fn encode_action(mv: &Move) -> usize {
    match mv {
        Move::Pass { .. } => pass_action(),
        Move::Place {
            orientation_global_id,
            x,
            y,
            ..
        } => orientation_global_id * BOARD_CELLS + y * BOARD_SIZE + x,
    }
}

pub fn decode_action(action: usize, player: usize) -> Option<Move> {
    if action == pass_action() {
        return Some(Move::Pass { player });
    }
    if action >= action_size() {
        return None;
    }
    let orientation_global_id = action / BOARD_CELLS;
    let position = action % BOARD_CELLS;
    let orientation = get_orientation(orientation_global_id)?;
    Some(Move::Place {
        player,
        piece_id: orientation.piece_id.clone(),
        orientation_global_id,
        x: position % BOARD_SIZE,
        y: position / BOARD_SIZE,
    })
}

fn cells_for_move(mv: &Move) -> Vec<(usize, usize)> {
    let Move::Place {
        orientation_global_id,
        x,
        y,
        ..
    } = mv
    else {
        return Vec::new();
    };
    let Some(orientation) = get_orientation(*orientation_global_id) else {
        return Vec::new();
    };
    orientation
        .cells
        .iter()
        .filter_map(|[dx, dy]| {
            let cx = *x as isize + dx;
            let cy = *y as isize + dy;
            in_bounds(cx, cy).then_some((cx as usize, cy as usize))
        })
        .collect()
}

fn start_point(point: StartPoint) -> (usize, usize) {
    match point {
        StartPoint::A => (0, 0),
        StartPoint::B => (13, 13),
    }
}

fn covers_start_point(cells: &[(usize, usize)], point: StartPoint) -> bool {
    let start = start_point(point);
    cells.iter().any(|cell| *cell == start)
}

fn required_start_point_for_first_move(state: &State, player: usize) -> Option<StartPoint> {
    if state.start_policy == StartPolicy::FixedStart {
        return Some(if player == 0 {
            StartPoint::A
        } else {
            StartPoint::B
        });
    }
    state.start_assignment[player]
}

pub fn is_legal_placement(state: &State, mv: &Move) -> bool {
    if state.status != GameStatus::Playing {
        return false;
    }
    let Move::Place {
        player,
        piece_id,
        orientation_global_id,
        ..
    } = mv
    else {
        return false;
    };
    if *player != state.current_player {
        return false;
    }
    if !state.remaining_pieces[*player]
        .iter()
        .any(|piece| piece == piece_id)
    {
        return false;
    }
    let Some(orientation) = get_orientation(*orientation_global_id) else {
        return false;
    };
    if orientation.piece_id != *piece_id {
        return false;
    }

    let raw_cell_count = orientation.cells.len();
    let cells = cells_for_move(mv);
    if cells.len() != raw_cell_count {
        return false;
    }
    for (x, y) in &cells {
        if state.board[cell_index(*x, *y)] != EMPTY {
            return false;
        }
    }

    let first_move = state.placed_pieces[*player].is_empty();
    if first_move {
        return match required_start_point_for_first_move(state, *player) {
            Some(point) => covers_start_point(&cells, point),
            None => {
                covers_start_point(&cells, StartPoint::A)
                    || covers_start_point(&cells, StartPoint::B)
            }
        };
    }

    for (x, y) in &cells {
        for (dx, dy) in ORTHOGONAL_DIRS {
            if get_cell(&state.board, *x as isize + dx, *y as isize + dy) == *player as i8 {
                return false;
            }
        }
    }

    cells.iter().any(|(x, y)| {
        DIAGONAL_DIRS.iter().any(|(dx, dy)| {
            get_cell(&state.board, *x as isize + dx, *y as isize + dy) == *player as i8
        })
    })
}

pub fn generate_legal_placements_for_player(state: &State, player: usize) -> Vec<Move> {
    if state.status != GameStatus::Playing {
        return Vec::new();
    }
    let mut scoped = state.clone();
    scoped.current_player = player;
    let mut moves = Vec::new();
    for piece_id in &scoped.remaining_pieces[player] {
        for orientation in get_orientations(piece_id) {
            for y in 0..BOARD_SIZE {
                for x in 0..BOARD_SIZE {
                    let mv = Move::Place {
                        player,
                        piece_id: piece_id.clone(),
                        orientation_global_id: orientation.global_id,
                        x,
                        y,
                    };
                    if is_legal_placement(&scoped, &mv) {
                        moves.push(mv);
                    }
                }
            }
        }
    }
    moves
}

pub fn generate_legal_moves(state: &State) -> Vec<Move> {
    if state.status != GameStatus::Playing {
        return Vec::new();
    }
    let placements = generate_legal_placements_for_player(state, state.current_player);
    if placements.is_empty() {
        vec![Move::Pass {
            player: state.current_player,
        }]
    } else {
        placements
    }
}

fn same_move(a: &Move, b: &Move) -> bool {
    match (a, b) {
        (Move::Pass { player: pa }, Move::Pass { player: pb }) => pa == pb,
        (
            Move::Place {
                player: pa,
                piece_id: piece_a,
                orientation_global_id: orientation_a,
                x: xa,
                y: ya,
            },
            Move::Place {
                player: pb,
                piece_id: piece_b,
                orientation_global_id: orientation_b,
                x: xb,
                y: yb,
            },
        ) => {
            pa == pb && piece_a == piece_b && orientation_a == orientation_b && xa == xb && ya == yb
        }
        _ => false,
    }
}

fn remove_piece(piece_list: &mut Vec<String>, piece_id: &str) {
    if let Some(index) = piece_list.iter().position(|piece| piece == piece_id) {
        piece_list.remove(index);
    }
}

fn update_start_assignment_after_first_move(
    state: &mut State,
    player: usize,
    cells: &[(usize, usize)],
) {
    if state.start_policy != StartPolicy::ChooseStart || state.start_assignment[player].is_some() {
        return;
    }
    let chosen = if covers_start_point(cells, StartPoint::A) {
        StartPoint::A
    } else {
        StartPoint::B
    };
    state.start_assignment[player] = Some(chosen);
    state.start_assignment[other(player)] = Some(if chosen == StartPoint::A {
        StartPoint::B
    } else {
        StartPoint::A
    });
}

fn has_legal_placement(state: &State, player: usize) -> bool {
    !generate_legal_placements_for_player(state, player).is_empty()
}

pub fn is_terminal(state: &State) -> bool {
    (state.remaining_pieces[0].is_empty() && state.remaining_pieces[1].is_empty())
        || (!has_legal_placement(state, 0) && !has_legal_placement(state, 1))
}

fn advance_turn_or_finish(state: &mut State) {
    if is_terminal(state) {
        state.status = GameStatus::Finished;
    } else {
        state.current_player = other(state.current_player);
        state.turn += 1;
    }
}

pub fn apply_move(state: &State, mv: &Move) -> Result<State, String> {
    if !generate_legal_moves(state)
        .iter()
        .any(|candidate| same_move(candidate, mv))
    {
        return Err("Illegal move".to_owned());
    }

    let mut next = state.clone();
    match mv {
        Move::Pass { player } => {
            next.forced_passed[*player] = true;
            next.consecutive_passes += 1;
            advance_turn_or_finish(&mut next);
        }
        Move::Place {
            player, piece_id, ..
        } => {
            let cells = cells_for_move(mv);
            for (x, y) in &cells {
                set_cell(&mut next.board, *x, *y, *player);
            }
            remove_piece(&mut next.remaining_pieces[*player], piece_id);
            next.placed_pieces[*player].push(piece_id.clone());
            next.last_placed_piece[*player] = Some(piece_id.clone());
            next.forced_passed[*player] = false;
            next.consecutive_passes = 0;
            if next.placed_pieces[*player].len() == 1 {
                update_start_assignment_after_first_move(&mut next, *player, &cells);
            }
            advance_turn_or_finish(&mut next);
        }
    }
    Ok(next)
}

pub fn piece_size(piece_id: &str) -> usize {
    orientations()
        .iter()
        .find(|orientation| orientation.piece_id == piece_id)
        .map(|orientation| orientation.unit_squares)
        .unwrap_or(0)
}

pub fn remaining_unit_squares(state: &State, player: usize) -> usize {
    state.remaining_pieces[player]
        .iter()
        .map(|piece| piece_size(piece))
        .sum()
}

pub fn score_player(state: &State, player: usize) -> isize {
    let remaining = remaining_unit_squares(state, player) as isize;
    let completed = state.remaining_pieces[player].is_empty();
    let completion_bonus = if completed { 15 } else { 0 };
    let monomino_last_bonus =
        if completed && state.last_placed_piece[player].as_deref() == Some("I1") {
            5
        } else {
            0
        };
    -remaining + completion_bonus + monomino_last_bonus
}

pub fn score_state(state: &State) -> [isize; 2] {
    [score_player(state, 0), score_player(state, 1)]
}

fn normalize_coord(player: usize, x: usize, y: usize) -> (usize, usize) {
    if player == 0 {
        (x, y)
    } else {
        (BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y)
    }
}

fn normalized_board_cell(state: &State, player: usize, x: usize, y: usize) -> i8 {
    let (nx, ny) = normalize_coord(player, x, y);
    state.board[cell_index(nx, ny)]
}

fn start_point_for_perspective(player: usize, owner: usize) -> (usize, usize) {
    let point = if owner == player {
        StartPoint::A
    } else {
        StartPoint::B
    };
    let (x, y) = start_point(point);
    if owner == 0 {
        (x, y)
    } else {
        (BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y)
    }
}

fn compute_corner_candidates(
    state: &State,
    perspective_player: usize,
    owner: usize,
) -> HashSet<(usize, usize)> {
    let mut cells = HashSet::new();
    if state.placed_pieces[owner].is_empty() {
        cells.insert(start_point_for_perspective(perspective_player, owner));
        return cells;
    }

    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            if normalized_board_cell(state, perspective_player, x, y) != owner as i8 {
                continue;
            }
            for (dx, dy) in DIAGONAL_DIRS {
                let cx = x as isize + dx;
                let cy = y as isize + dy;
                if !in_bounds(cx, cy) {
                    continue;
                }
                let cx = cx as usize;
                let cy = cy as usize;
                if normalized_board_cell(state, perspective_player, cx, cy) != EMPTY {
                    continue;
                }
                let edge_blocked = ORTHOGONAL_DIRS.iter().any(|(ex, ey)| {
                    let nx = cx as isize + ex;
                    let ny = cy as isize + ey;
                    in_bounds(nx, ny)
                        && normalized_board_cell(
                            state,
                            perspective_player,
                            nx as usize,
                            ny as usize,
                        ) == owner as i8
                });
                if !edge_blocked {
                    cells.insert((cx, cy));
                }
            }
        }
    }
    cells
}

fn compute_forbidden_edge_cells(
    state: &State,
    perspective_player: usize,
    owner: usize,
) -> HashSet<(usize, usize)> {
    let mut cells = HashSet::new();
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            if normalized_board_cell(state, perspective_player, x, y) != owner as i8 {
                continue;
            }
            for (dx, dy) in ORTHOGONAL_DIRS {
                let cx = x as isize + dx;
                let cy = y as isize + dy;
                if in_bounds(cx, cy)
                    && normalized_board_cell(state, perspective_player, cx as usize, cy as usize)
                        == EMPTY
                {
                    cells.insert((cx as usize, cy as usize));
                }
            }
        }
    }
    cells
}

pub fn encode_state_tensor(state: &State, player: usize) -> Vec<f32> {
    let mut planes = vec![0.0; STATE_PLANES * BOARD_CELLS];
    let opponent = other(player);
    let my_corners = compute_corner_candidates(state, player, player);
    let opp_corners = compute_corner_candidates(state, player, opponent);
    let my_forbidden = compute_forbidden_edge_cells(state, player, player);
    let opp_forbidden = compute_forbidden_edge_cells(state, player, opponent);
    let my_start = start_point_for_perspective(player, player);
    let opp_start = start_point_for_perspective(player, opponent);

    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let index = y * BOARD_SIZE + x;
            let cell = normalized_board_cell(state, player, x, y);
            let setters = [
                cell == player as i8,
                cell == opponent as i8,
                cell == EMPTY,
                my_corners.contains(&(x, y)),
                opp_corners.contains(&(x, y)),
                my_forbidden.contains(&(x, y)),
                opp_forbidden.contains(&(x, y)),
                (x, y) == my_start,
                (x, y) == opp_start,
            ];
            for (plane, enabled) in setters.iter().enumerate() {
                if *enabled {
                    planes[plane * BOARD_CELLS + index] = 1.0;
                }
            }
            for (piece_index, piece_id) in PIECE_IDS.iter().enumerate() {
                if state.remaining_pieces[player]
                    .iter()
                    .any(|piece| piece == piece_id)
                {
                    planes[(9 + piece_index) * BOARD_CELLS + index] = 1.0;
                }
                if state.remaining_pieces[opponent]
                    .iter()
                    .any(|piece| piece == piece_id)
                {
                    planes[(30 + piece_index) * BOARD_CELLS + index] = 1.0;
                }
            }
        }
    }
    planes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_and_action_counts_match_js_core() {
        assert_eq!(PIECE_IDS.len(), 21);
        assert_eq!(orientation_count(), 91);
        assert_eq!(pass_action(), 91 * 14 * 14);
        assert_eq!(action_size(), 91 * 14 * 14 + 1);
    }

    #[test]
    fn initial_legal_move_counts_match_js_core() {
        let choose = create_initial_state(StartPolicy::ChooseStart);
        let fixed = create_initial_state(StartPolicy::FixedStart);
        assert_eq!(generate_legal_moves(&choose).len(), 116);
        assert_eq!(generate_legal_moves(&fixed).len(), 58);
    }

    #[test]
    fn action_encode_decode_round_trips() {
        let state = create_initial_state(StartPolicy::FixedStart);
        for mv in generate_legal_moves(&state) {
            let action = encode_action(&mv);
            assert_eq!(decode_action(action, state.current_player), Some(mv));
        }
        assert_eq!(
            decode_action(pass_action(), 1),
            Some(Move::Pass { player: 1 })
        );
    }

    #[test]
    fn apply_move_and_score_match_initial_expectations() {
        let state = create_initial_state(StartPolicy::FixedStart);
        assert_eq!(score_state(&state), [-89, -89]);
        let first = generate_legal_moves(&state)[0].clone();
        let next = apply_move(&state, &first).expect("first move should be legal");
        assert_eq!(next.current_player, 1);
        assert_eq!(next.turn, 1);
        assert_eq!(next.remaining_pieces[0].len(), 20);
        assert_eq!(
            encode_state_tensor(&next, 0).len(),
            STATE_PLANES * BOARD_CELLS
        );
    }
}
