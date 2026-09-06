# knn-image-classifier — Agent Guide

## Overview

A zero-dependency Rust demo that implements a k-nearest-neighbors image classifier
using only the standard library. It generates synthetic 5×5 grayscale images for
four pattern classes and evaluates k-NN accuracy across multiple distance metrics
and k values.

## Project Layout

| Path | Purpose |
|------|---------|
| `src/main.rs` | Entire demo: data generation, k-NN, evaluation, CLI |
| `examples/` | Additional runnable examples (if any) |
| `Cargo.toml` | Single-crate package manifest |

## Build / Run

```bash
cargo run --release
```

The binary prints accuracy tables for Euclidean, Manhattan, cosine, and custom
distance metrics over k = 1, 3, 5, 7, 9.

## Test / Lint

```bash
# Run the small built-in test suite
cargo test --release

# Format check
cargo fmt -- --check

# Clippy
cargo clippy -- -D warnings
```

## Key Conventions

- **stdlib only**: no external crates.
- **Fixed 5×5 images**: represented as `[u8; 25]`.
- **Synthetic classes**: horizontal line, vertical line, diagonal, checkerboard.
- **Distance metrics**: Euclidean, Manhattan, cosine, plus optional user-defined metric.

## Common Gotchas

- The `.cargo/config.toml` points `target-dir` to a shared `/.cargo-targets` path.
  If that directory is unavailable, override with `CARGO_TARGET_DIR=/tmp/...`.
- `cargo fmt --check` may report formatting issues; the project currently uses
  manual formatting in some regions.
- Build artifacts are tiny, but the shared target dir may be read-only in some
  environments.

## Deployment

No CI workflow. This is a learning/demo binary; run locally with `cargo run`.
