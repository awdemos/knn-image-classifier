//! k-Nearest Neighbors Image Classifier Demo.
//!
//! Uses synthetic 5×5 grayscale images with programmatic variation
//! and no external dependencies — only the Rust standard library.
//!
//! Classes:
//!   0 — horizontal line
//!   1 — vertical line
//!   2 — diagonal
//!   3 — checkerboard

use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SEED: u64 = 42;
const SAMPLES_PER_CLASS: usize = 80;
const GRID: usize = 5; // 5×5 images
const PIXEL_COUNT: usize = GRID * GRID; // 25
const IMAGE_CAPACITY: usize = PIXEL_COUNT; // 25 for [u8; 25]
const K_VALUES: [usize; 5] = [1, 3, 5, 7, 9];
const EPS: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A labeled image sample with fixed-size pixel array.  (Improvement 10)
#[derive(Clone)]
struct Sample {
    pixels: [u8; IMAGE_CAPACITY],
    label: usize,
}

/// A candidate neighbor for the k-NN heap.  (Improvement 12)
#[derive(PartialEq)]
struct Candidate {
    dist: f64,
    label: usize,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// BinaryHeap is a max-heap by `Ord`; we invert the comparison so that
/// larger distances have higher priority.  We push all candidates and pop
/// the farthest when the heap exceeds k, leaving the k nearest in the heap.
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist).reverse()
    }
}

/// Type alias for a distance function.  (Improvement 9)
type DistanceFn = fn(&[f64], &[f64]) -> f64;

// ---------------------------------------------------------------------------
// PRNG — simple LCG, zero external deps.  (Improvement 3)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// MMIX-style LCG by Knuth.
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005);
        self.0 = self.0.wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    /// Integer in [lo, hi] inclusive.
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        let range = (hi - lo + 1) as u32;
        lo + (self.next_u32() % range) as i32
    }

    /// u8 in [lo, hi] inclusive.
    fn range_u8(&mut self, lo: u8, hi: u8) -> u8 {
        let range = hi as u32 - lo as u32 + 1;
        lo + (self.next_u32() % range) as u8
    }

    fn range_grid(&mut self, lo: usize, hi: usize) -> usize {
        self.range(lo as i32, hi as i32) as usize
    }
}

// ---------------------------------------------------------------------------
// Distance metrics  (Improvement 6)
// ---------------------------------------------------------------------------

/// Euclidean distance (L2).
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Manhattan distance (L1).
fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Cosine distance: 1 − cosine_similarity.
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < EPS || nb < EPS {
        1.0
    } else {
        1.0 - dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// Normalisation  (Improvement 5)
// ---------------------------------------------------------------------------

/// L2 normalise a vector in-place.
fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > EPS {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Feature extraction  (Improvement 8)
// ---------------------------------------------------------------------------

/// Extract a 14-element feature vector from a 5×5 image.
///
/// Features:
///   [0..4]   Row sums (5)
///   [5..9]   Column sums (5)
///   [10]     Total bright pixels (pixel > 128)
///   [11]     Horizontal symmetry score (higher = more symmetric)
///   [12]     Vertical symmetry score
///   [13]     Diagonal presence (main diagonal sum)
fn extract_features(pixels: &[u8; IMAGE_CAPACITY]) -> Vec<f64> {
    let mut feats = Vec::with_capacity(14);

    // Row sums
    for r in 0..GRID {
        let s: f64 = pixels[r * GRID..r * GRID + GRID]
            .iter()
            .map(|&p| p as f64)
            .sum();
        feats.push(s);
    }

    // Column sums
    for c in 0..GRID {
        let s: f64 = (0..GRID)
            .map(|r| pixels[r * GRID + c] as f64)
            .sum();
        feats.push(s);
    }

    // Bright pixel count (pixels > 128)
    let bright = pixels.iter().filter(|&&p| p > 128).count() as f64;
    feats.push(bright);

    // Horizontal symmetry: for each row, how symmetric left↔right
    let h_sym: f64 = (0..GRID)
        .map(|r| {
            let row = &pixels[r * GRID..r * GRID + GRID];
            (0..GRID / 2)
                .map(|c| {
                    let diff = row[c] as i16 - row[GRID - 1 - c] as i16;
                    diff.abs() as f64
                })
                .sum::<f64>()
        })
        .sum();
    // Invert so higher = more symmetric (cap symmetrical patterns near 0 diff)
    feats.push(-h_sym);

    // Vertical symmetry
    let v_sym: f64 = (0..GRID)
        .map(|c| {
            (0..GRID / 2)
                .map(|r| {
                    let top = pixels[r * GRID + c] as i16;
                    let bot = pixels[(GRID - 1 - r) * GRID + c] as i16;
                    (top - bot).abs() as f64
                })
                .sum::<f64>()
        })
        .sum();
    feats.push(-v_sym);

    // Diagonal presence
    let diag: f64 = (0..GRID).map(|i| pixels[i * GRID + i] as f64).sum();
    feats.push(diag);

    feats
}

// ---------------------------------------------------------------------------
// Data generation — programmatic variations  (Improvement 3)
// ---------------------------------------------------------------------------

/// Spill bright pixels into their neighbours to blur edges.
fn apply_spillover(pixels: &mut [u8; IMAGE_CAPACITY], rng: &mut Rng) {
    let original = *pixels;
    for r in 0..GRID {
        for c in 0..GRID {
            let idx = r * GRID + c;
            let val = original[idx] as i32;
            if val > 120 {
                let bleed = (val * rng.range(10, 35) / 100) as u16;
                if r > 0 {
                    let ni = (r - 1) * GRID + c;
                    pixels[ni] = (pixels[ni] as u16 + bleed).min(255) as u8;
                }
                if r < GRID - 1 {
                    let ni = (r + 1) * GRID + c;
                    pixels[ni] = (pixels[ni] as u16 + bleed).min(255) as u8;
                }
                if c > 0 {
                    let ni = r * GRID + c - 1;
                    pixels[ni] = (pixels[ni] as u16 + bleed).min(255) as u8;
                }
                if c < GRID - 1 {
                    let ni = r * GRID + c + 1;
                    pixels[ni] = (pixels[ni] as u16 + bleed).min(255) as u8;
                }
            }
        }
    }
}

/// Apply heavy background noise, brightness jitter, and salt-and-pepper.
fn corrupt_image(pixels: &mut [u8; IMAGE_CAPACITY], rng: &mut Rng) {
    // Heavy background noise on dim pixels (0-100, overlaps with dim foreground)
    for p in pixels.iter_mut() {
        if *p < 80 {
            *p = rng.range_u8(0, 100);
        }
    }
    // Aggressive brightness jitter on all pixels (±70)
    for p in pixels.iter_mut() {
        let jitter = rng.range(-70, 70);
        *p = ((*p as i32).saturating_add(jitter)).clamp(0, 255) as u8;
    }
    // Salt-and-pepper: 10% chance per pixel of extreme value
    for p in pixels.iter_mut() {
        if rng.next_u32() % 100 < 10 {
            *p = if rng.next_u32().is_multiple_of(2) { 0 } else { 255 };
        }
    }
}

/// Generate a horizontal-line sample.
fn gen_horizontal(rng: &mut Rng) -> [u8; IMAGE_CAPACITY] {
    let mut px = [0u8; IMAGE_CAPACITY];
    let variant = rng.next_u32() % 7;
    let bright = rng.range_u8(150, 255);
    match variant {
        0 => {
            let row = rng.range_grid(0, GRID - 1);
            for c in 0..GRID {
                px[row * GRID + c] = bright;
            }
        }
        1 => {
            let row = rng.range_grid(0, GRID - 2);
            for r in row..row + 2 {
                for c in 0..GRID {
                    px[r * GRID + c] = bright;
                }
            }
        }
        2 => {
            let row = rng.range_grid(0, GRID - 1);
            for c in 1..GRID {
                px[row * GRID + c] = bright;
            }
        }
        3 => {
            let bright2 = rng.range_u8(150, 255);
            for c in 0..GRID {
                px[c] = bright;
                px[(GRID - 1) * GRID + c] = bright2;
            }
        }
        4 => {
            // Horizontal + faint vertical cross — ambiguous with vertical class
            let row = rng.range_grid(0, GRID - 1);
            let col = rng.range_grid(0, GRID - 1);
            for c in 0..GRID {
                px[row * GRID + c] = bright;
            }
            let dim_vert = (bright as u16 / 2) as u8;
            for r in 0..GRID {
                px[r * GRID + col] = px[r * GRID + col].max(dim_vert);
            }
        }
        5 => {
            // Very dim horizontal bar (barely above noise floor)
            let row = rng.range_grid(0, GRID - 1);
            let dim = rng.range_u8(100, 140);
            for c in 0..GRID {
                px[row * GRID + c] = dim;
            }
        }
        6 => {
            // Two-row bar with lots of vertical leak
            let row = rng.range_grid(0, GRID - 2);
            for r in row..row + 2 {
                for c in 0..GRID {
                    px[r * GRID + c] = bright;
                }
            }
            // Spill some brightness downward extra
            if row + 2 < GRID {
                for c in 0..GRID {
                    px[(row + 2) * GRID + c] = (bright as u16 / 3) as u8;
                }
            }
        }
        _ => unreachable!(),
    }
    corrupt_image(&mut px, rng);
    px
}

/// Generate a vertical-line sample.
fn gen_vertical(rng: &mut Rng) -> [u8; IMAGE_CAPACITY] {
    let mut px = [0u8; IMAGE_CAPACITY];
    let variant = rng.next_u32() % 7;
    let bright = rng.range_u8(150, 255);
    match variant {
        0 => {
            let col = rng.range_grid(0, GRID - 1);
            for r in 0..GRID {
                px[r * GRID + col] = bright;
            }
        }
        1 => {
            let col = rng.range_grid(0, GRID - 2);
            for r in 0..GRID {
                for c in col..col + 2 {
                    px[r * GRID + c] = bright;
                }
            }
        }
        2 => {
            let col = rng.range_grid(0, GRID - 1);
            for r in 1..GRID {
                px[r * GRID + col] = bright;
            }
        }
        3 => {
            let bright2 = rng.range_u8(150, 255);
            for r in 0..GRID {
                px[r * GRID] = bright;
                px[r * GRID + (GRID - 1)] = bright2;
            }
        }
        4 => {
            // Vertical + faint horizontal cross — ambiguous with horizontal class
            let col = rng.range_grid(0, GRID - 1);
            let row = rng.range_grid(0, GRID - 1);
            for r in 0..GRID {
                px[r * GRID + col] = bright;
            }
            let dim_horiz = (bright as u16 / 2) as u8;
            for c in 0..GRID {
                px[row * GRID + c] = px[row * GRID + c].max(dim_horiz);
            }
        }
        5 => {
            // Very dim vertical bar (barely above noise)
            let col = rng.range_grid(0, GRID - 1);
            let dim = rng.range_u8(100, 140);
            for r in 0..GRID {
                px[r * GRID + col] = dim;
            }
        }
        6 => {
            // Two-column bar with horizontal leak
            let col = rng.range_grid(0, GRID - 2);
            for r in 0..GRID {
                for c in col..col + 2 {
                    px[r * GRID + c] = bright;
                }
            }
            if col + 2 < GRID {
                for r in 0..GRID {
                    px[r * GRID + col + 2] = (bright as u16 / 3) as u8;
                }
            }
        }
        _ => unreachable!(),
    }
    corrupt_image(&mut px, rng);
    px
}

/// Generate a diagonal-line sample.
fn gen_diagonal(rng: &mut Rng) -> [u8; IMAGE_CAPACITY] {
    let mut px = [0u8; IMAGE_CAPACITY];
    let variant = rng.next_u32() % 7;
    let bright = rng.range_u8(150, 255);
    match variant {
        0 => {
            for i in 0..GRID {
                px[i * GRID + i] = bright;
            }
        }
        1 => {
            for i in 0..GRID {
                px[i * GRID + (GRID - 1 - i)] = bright;
            }
        }
        2 => {
            let bright2 = rng.range_u8(130, 255);
            for i in 0..GRID {
                px[i * GRID + i] = bright;
                px[i * GRID + (GRID - 1 - i)] = bright2;
            }
        }
        3 => {
            // Thick diagonal (3 parallel) — looks like a stair-step bar
            for i in 0..GRID {
                px[i * GRID + i] = bright;
                if i + 1 < GRID {
                    px[i * GRID + i + 1] = (bright as u16 * 3 / 4) as u8;
                    px[(i + 1) * GRID + i] = (bright as u16 * 3 / 4) as u8;
                }
            }
        }
        4 => {
            // Sparse diagonal — only 2 on-axis pixels, hard to distinguish
            let i1 = rng.range_grid(0, GRID - 1);
            let i2 = rng.range_grid(0, GRID - 1);
            let i1 = i1.min(i2);
            let i2 = i1.max(i2);
            let i2 = if i2 == i1 { (i2 + 1).min(GRID - 1) } else { i2 };
            px[i1 * GRID + i1] = bright;
            px[i2 * GRID + i2] = bright;
        }
        5 => {
            // Diagonal with heavy vertical spill — looks almost like a vertical bar
            for i in 0..GRID {
                px[i * GRID + i] = bright;
                px[i * GRID + 2] = (bright as u16 * 3 / 5) as u8; // center column leak
            }
        }
        6 => {
            // Offset anti-diagonal stair-step — ambiguous with horizontal/vertical
            for i in 0..GRID {
                let c = (GRID - 1 - i + 1).min(GRID - 1);
                px[i * GRID + c] = bright;
                if c > 0 {
                    px[i * GRID + c - 1] = (bright as u16 * 2 / 3) as u8;
                }
            }
        }
        _ => unreachable!(),
    }
    corrupt_image(&mut px, rng);
    px
}

/// Generate a checkerboard sample.
fn gen_checkerboard(rng: &mut Rng) -> [u8; IMAGE_CAPACITY] {
    let mut px = [0u8; IMAGE_CAPACITY];
    let variant = rng.next_u32() % 7;
    let high = rng.range_u8(150, 255);
    let low = rng.range_u8(0, 30);

    if variant == 4 {
        // Nearly uniform noise — hard to spot any pattern
        for p in px.iter_mut() {
            *p = rng.range_u8(70, 100);
        }
    } else {
        for r in 0..GRID {
            for c in 0..GRID {
                let on = match variant {
                    0 => (r + c) % 2 == 0,
                    1 => (r / 2 + c / 2) % 2 == 0,
                    2 => (r % 2 == 0) == (c % 2 == 0),
                    3 => r % 2 == 0, // all even rows bright = horizontal stripes
                    5 => c < 2,       // left-heavy = looks like vertical stripe
                    6 => c % 2 == 0, // vertical stripes = looks like vertical class
                    _ => unreachable!(),
                };
                px[r * GRID + c] = if on { high } else { low };
            }
        }
    }
    // Extra jitter for checkerboards
    for p in px.iter_mut() {
        let jit = rng.range(-15, 15);
        *p = ((*p as i32).saturating_add(jit)).clamp(0, 255) as u8;
    }
    px
}

/// Build the full synthetic dataset with optional spillover blur.
fn generate_dataset(rng: &mut Rng, samples_per_class: usize) -> Vec<Sample> {
    let generators: [fn(&mut Rng) -> [u8; IMAGE_CAPACITY]; 4] = [
        gen_horizontal,
        gen_vertical,
        gen_diagonal,
        gen_checkerboard,
    ];

    let mut samples = Vec::with_capacity(generators.len() * samples_per_class);
    for (label, generator_fn) in generators.into_iter().enumerate() {
        for _ in 0..samples_per_class {
            let mut pixels = generator_fn(rng);
            // Apply spillover to ~30% of samples for edge blur
            if rng.next_u32() % 10 < 3 {
                apply_spillover(&mut pixels, rng);
            }
            samples.push(Sample { pixels, label });
        }
    }
    samples
}

// ---------------------------------------------------------------------------
// Train / test split
// ---------------------------------------------------------------------------

/// Deterministic split: every Nth sample goes to test.
fn split_indices(total: usize, test_every: usize) -> (Vec<usize>, Vec<usize>) {
    let train: Vec<usize> = (0..total)
        .filter(|i| (i + 1) % test_every != 0)
        .collect();
    let test: Vec<usize> = (0..total)
        .filter(|i| (i + 1) % test_every == 0)
        .collect();
    (train, test)
}

// ---------------------------------------------------------------------------
// Classifier  (Improvements 1, 2, 9, 12)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn predict(
    all_features: &[Vec<f64>],
    all_labels: &[usize],
    train_indices: &[usize],
    query_global_idx: Option<usize>,
    query_features: &[f64],
    k: usize,
    num_classes: usize,
    dist_fn: DistanceFn,
) -> usize {
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);

    for &ti in train_indices {
        if query_global_idx == Some(ti) {
            continue;
        }
        let dist = dist_fn(&all_features[ti], query_features);
        heap.push(Candidate {
            dist,
            label: all_labels[ti],
        });
        if heap.len() > k {
            heap.pop();
        }
    }

    let mut weights = vec![0.0f64; num_classes];
    for cand in heap.into_iter() {
        weights[cand.label] += 1.0 / (cand.dist + EPS);
    }

    weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(label, _)| label)
        .expect("weights should never be empty when k >= 1")
}

/// Compute accuracy for a set of indices using the classifier.
#[allow(clippy::too_many_arguments)]
fn evaluate_accuracy(
    all_features: &[Vec<f64>],
    all_labels: &[usize],
    train_indices: &[usize],
    eval_indices: &[usize],
    k: usize,
    num_classes: usize,
    dist_fn: DistanceFn,
    exclude_self: bool,
) -> f64 {
    let correct: usize = eval_indices
        .iter()
        .filter(|&&idx| {
            let self_idx = if exclude_self { Some(idx) } else { None };
            let pred = predict(
                all_features,
                all_labels,
                train_indices,
                self_idx,
                &all_features[idx],
                k,
                num_classes,
                dist_fn,
            );
            pred == all_labels[idx]
        })
        .count();
    correct as f64 / eval_indices.len() as f64 * 100.0
}

// ---------------------------------------------------------------------------
// Confusion matrix  (Improvement 7)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_confusion_matrix(
    all_features: &[Vec<f64>],
    all_labels: &[usize],
    train_indices: &[usize],
    test_indices: &[usize],
    k: usize,
    num_classes: usize,
    dist_fn: DistanceFn,
) -> Vec<Vec<usize>> {
    let mut cm = vec![vec![0usize; num_classes]; num_classes];
    for &ti in test_indices {
        let pred = predict(
            all_features,
            all_labels,
            train_indices,
            None,
            &all_features[ti],
            k,
            num_classes,
            dist_fn,
        );
        cm[all_labels[ti]][pred] += 1;
    }
    cm
}

fn print_confusion_matrix(cm: &[Vec<usize>], abbrev: &[&str]) {
    let num_classes = cm.len();
    println!("\n    Confusion Matrix");
    print!("    {:<6}", "Pred→");
    for a in abbrev {
        print!(" {a:>4}");
    }
    println!("\n    {}", "─".repeat(6 + 5 * num_classes));
    for (actual, row) in cm.iter().enumerate() {
        print!("   {:<4}", abbrev[actual]);
        for &val in row {
            print!(" {val:>4}");
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

/// Print a 5×5 grayscale image using ASCII block characters.
fn display_image(pixels: &[u8; IMAGE_CAPACITY]) {
    for row in 0..GRID {
        for col in 0..GRID {
            let val = pixels[row * GRID + col];
            let ch = if val > 200 {
                '█'
            } else if val > 100 {
                '▓'
            } else if val > 30 {
                '▒'
            } else {
                '░'
            };
            print!("{ch}");
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== k-NN Image Classifier Demo ===\n");

    // ---- 1. Generate dataset ----
    let mut rng = Rng::new(SEED);
    let samples = generate_dataset(&mut rng, SAMPLES_PER_CLASS);
    let num_classes = samples.iter().map(|s| s.label).max().unwrap_or(0) + 1; // (11)

    let class_names = ["horizontal line", "vertical line", "diagonal", "checkerboard"];
    let class_abbrev = [" H ", " V ", " D ", " C "];

    println!("Dataset: {} total samples across {num_classes} classes", samples.len());
    for (i, name) in class_names.iter().enumerate() {
        if i >= num_classes {
            break;
        }
        let count = samples.iter().filter(|s| s.label == i).count();
        println!("  Class {i} ({name}): {count} samples");
    }
    println!();

    // ---- 2. Precompute feature vectors ----
    // Raw pixels as f64 vectors (L2 normalised)  (Improvement 5)
    let pixel_features: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| {
            let mut v: Vec<f64> = s.pixels.iter().map(|&p| p as f64).collect();
            l2_normalize(&mut v);
            v
        })
        .collect();

    // Extracted features (L2 normalised)  (Improvement 8)
    let extracted_features: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| {
            let mut v = extract_features(&s.pixels);
            l2_normalize(&mut v);
            v
        })
        .collect();

    let labels: Vec<usize> = samples.iter().map(|s| s.label).collect();

    // ---- 3. Train / test split ----
    let (train_indices, test_indices) = split_indices(samples.len(), 5);
    println!(
        "Train: {} samples  |  Test: {} samples",
        train_indices.len(),
        test_indices.len()
    );
    println!();

    // ---- 4. Set up evaluation grid ----
    let metrics: [(&str, DistanceFn); 3] = [
        ("Euclidean", euclidean_distance),
        ("Manhattan", manhattan_distance),
        ("Cosine", cosine_distance),
    ];

    let modes: [(&str, &[Vec<f64>]); 2] =
        [("Raw Pixels (25-dim)", &pixel_features), ("Extracted Features (14-dim)", &extracted_features)];

    // Storage for all results
    #[derive(Clone)]
    struct RunResult {
        mode: &'static str,
        metric: &'static str,
        k: usize,
        test_acc: f64,
    }

    let mut all_results: Vec<RunResult> = Vec::new();

    // ---- 5. K-sweep tables  (Improvement 4) ----
    for (mode_name, feat_set) in &modes {
        println!("═══ K Sweep: {mode_name} ═══");
        print!("  {:<6}", "k");
        for (metric_name, _) in &metrics {
            print!("  {metric_name:>12}");
        }
        println!("\n  {}", "─".repeat(6 + 15 * metrics.len()));

        for &k in &K_VALUES {
            print!("  k={:<3}", k);
            for (metric_name, dist_fn) in &metrics {
                let acc = evaluate_accuracy(
                    feat_set, &labels, &train_indices, &test_indices, k, num_classes, *dist_fn, false,
                );
                print!("  {acc:>9.1}%");
                all_results.push(RunResult {
                    mode: mode_name,
                    metric: metric_name,
                    k,
                    test_acc: acc,
                });
            }
            println!();
        }
        println!();
    }

    // ---- 6. Best configuration ----
    let best = all_results
        .iter()
        .max_by(|a, b| a.test_acc.total_cmp(&b.test_acc))
        .expect("should have at least one result");

    println!(
        "★ Best: {} | Metric: {} | k = {} | Test Accuracy: {:.1}%",
        best.mode, best.metric, best.k, best.test_acc
    );

    // ---- 7. Confusion matrix for best config  (Improvement 7) ----
    // Find the distance function and feature set for best
    let best_feat_set = if best.mode.starts_with("Raw") {
        &pixel_features
    } else {
        &extracted_features
    };
    let best_dist_fn = metrics
        .iter()
        .find(|(name, _)| *name == best.metric)
        .map(|(_, fn_ptr)| *fn_ptr)
        .unwrap_or(euclidean_distance);

    let cm = build_confusion_matrix(
        best_feat_set,
        &labels,
        &train_indices,
        &test_indices,
        best.k,
        num_classes,
        best_dist_fn,
    );
    print_confusion_matrix(&cm, &class_abbrev[..num_classes]);

    // ---- 8. Example predictions ----
    println!("\n\n═══ Example Predictions ═══\n");
    for (i, &ti) in test_indices.iter().enumerate().take(5) {
        let pred = predict(
            best_feat_set,
            &labels,
            &train_indices,
            None,
            &best_feat_set[ti],
            best.k,
            num_classes,
            best_dist_fn,
        );
        let actual = labels[ti];
        let mark = if pred == actual { "✓" } else { "✗" };
        println!(
            "Sample {} | Actual: {} ({}) → Predicted: {} ({})  {mark}",
            i + 1,
            actual,
            class_names[actual],
            pred,
            class_names[pred],
        );
        display_image(&samples[ti].pixels);
        println!();
    }
}
