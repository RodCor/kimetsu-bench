# Contributing to kimetsu-bench

Thanks for helping improve the kimetsu benchmark harness. This repo ships **how
to run** the benchmarks — the drivers, converters, and docs — never datasets or
raw results.

## Layout & building

`kimetsu-bench` is a standalone Rust project that path-depends on the
[kimetsu](https://github.com/RodCor/kimetsu) crates at `../crates`. Clone it
inside a kimetsu checkout (as `./bench/`) so `../crates/*` resolves, then:

```bash
cargo build            # kbench + kstress
cargo build --release  # for real benchmark runs
```

See the [README](README.md) for how to run each benchmark.

## The quality bar

Keep it green locally before opening a PR — the same bar kimetsu holds:

```bash
cargo fmt --all --check
cargo clippy --all-targets      # aim to keep it quiet
cargo test                      # driver unit tests
```

## What NEVER gets committed

- **Datasets and raw results.** They live under the gitignored `local/` folder;
  the drivers download/convert data and write reports there. Ship the *method*
  (drivers, converters, docs), not the data or the numbers.
- **Secrets.** `.env`, credentials, API keys, or brain exports. The auth code
  reads your *own* local tokens — never commit them. The brain redacts known
  token shapes, but treat these as never-commit regardless.

## Pull requests

- Keep PRs focused; say what changed and why.
- A new benchmark driver drops into `src/drivers/` and registers in
  `src/drivers/mod.rs` + `src/main.rs`.
- Report security issues privately (see the [Code of Conduct](CODE_OF_CONDUCT.md)
  contact) rather than in a public issue.

By contributing, you agree that your work is dual-licensed under the project's
terms (MIT OR Apache-2.0) and that you will abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
