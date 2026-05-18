# bedrockformation

A high-performance Rust port of [Developer-Mike's Minecraft Bedrock Formation Finder](https://github.com/Developer-Mike/Minecraft-Bedrock-Formation-Finder-1.18) for Minecraft Java Edition 1.18+.

Given a world seed and a set of observed bedrock blocks, this tool searches the world in a spiral outward from a given coordinate until it finds where that exact formation exists, letting you determine your coordinates in a seed purely from the shape of the bedrock layer.

---

## Table of contents

- [How it works](#how-it-works)
- [Differences from the Java original](#differences-from-the-java-original)
  - [Bug fixes](#bug-fixes)
  - [Performance improvements](#performance-improvements)
- [Building](#building)
- [Usage](#usage)
  - [Arguments](#arguments)
  - [Examples](#examples)
  - [Reading the bedrock layer](#reading-the-bedrock-layer)
- [Benchmarks](#benchmarks)
- [Credits](#credits)

---

## How it works

In Minecraft 1.18+, bedrock generation is deterministic: each block position is either bedrock or air based on the world seed, the block's XYZ coordinates, and which layer it sits in (floor or roof). The probability of a block being bedrock decreases linearly as you move away from the solid edge:

**Floor** (Y = −64 to −59)

| Y | Probability of bedrock |
|---|---|
| −64 | 100% (always bedrock) |
| −63 | 80% |
| −62 | 60% |
| −61 | 40% |
| −60 | 20% |
| −59 | 0% (never bedrock) |

**Roof** (Y = 122 to 127)

| Y | Probability of bedrock |
|---|---|
| 127 | 100% (always bedrock) |
| 126 | 80% |
| 125 | 60% |
| 124 | 40% |
| 123 | 20% |
| 122 | 0% (never bedrock) |

The tool encodes a formation as a list of offsets relative to an unknown origin (blocks that should be bedrock, and blocks that should be air), then walks a spiral of candidate origin coordinates, testing each one against the world seed until it finds a match.

The search uses the same RNG chain as Minecraft itself: the world seed is fed through SplitMix64 and Xoroshiro128++ to derive a per-layer deriver seed, which is then hashed per-block using `MathHelper.hashCode` to produce a single float that is compared against the block's probability threshold.

---

## Differences from the Java original

### Bug fixes

#### 1. `math_hash`: wrong integer width for large coordinates

Java evaluates `(long)(x * 3129871)` as a **32-bit** multiply that wraps before being sign-extended to 64 bits. The original port cast `x` to `i64` before multiplying, producing wrong results for any `|x| > ~688` where 32-bit overflow would have occurred.

```rust
// Wrong: multiplies in 64-bit, no overflow
let term_x = (x as i64) * 3_129_871;

// Correct: wraps in 32-bit, then sign-extends, matching Java
let term_x = x.wrapping_mul(3_129_871) as i64;
```

This was a silent correctness bug: the tool would find a result, but not the right one, for any coordinate outside a small central region.

#### 2. Thread pool: per-batch allocation

The previous port wrapped each batch's coordinate arrays in `Arc<Vec<i32>>`, which allocated and copied 512 KB of data on every 65 536-position chunk. Since the main thread blocks synchronously until all workers finish, the underlying arrays are stable for the entire batch, so no copy is needed.

---

### Performance improvements

| Change | Detail |
|---|---|
| **rayon replaces the custom thread pool** | Eliminated ~70 lines of `mpsc` channel plumbing. `find_first` handles early exit, empty-range dispatching, and spiral-order correctness automatically. |
| **No per-batch heap allocation** | The 512 KB `Vec` copy per chunk is gone entirely; workers borrow the chunk buffer directly through rayon's scoped parallelism. |
| **Cross-worker early exit** | As soon as any worker finds a match, rayon signals the others to stop. The custom pool had no cancellation mechanism. |
| **`sort_by_cached_key`** | Sort keys are computed exactly once per block. The previous version recomputed them on every comparison; an intermediate version built a `Vec<(f64, Block)>` and stripped it back out (two extra allocations and two extra passes). |
| **AoS chunk buffer** | Two parallel `Vec<i32>` (`chunk_x`, `chunk_z`) replaced by one `Vec<(i32, i32)>`. Each position's data is now adjacent in memory, reducing cache pressure in the hot loop. |
| **Dead variable removed** | The `filled` counter was always equal to `CHUNK_SIZE` when passed to the search, so the loop body always ran to completion. Removed; `find_first` now ranges directly over `0..CHUNK_SIZE`. |
| **Blocks immutable after sort** | The `mut` on `blocks` is shadowed away after sorting so the compiler enforces that nothing downstream can accidentally mutate the search list. |
| **Trivially-informationless blocks filtered** | Blocks whose outcome is guaranteed (always-bedrock declared as bedrock, or never-bedrock declared as air) are removed before the search. They pass every check trivially and contribute nothing to candidate rejection. |

---

## Building

Requires [Rust](https://rustup.rs) (stable, 1.70+).

```bash
git clone https://github.com/<you>/bedrockformation-rs
cd bedrockformation-rs
cargo build --release
```

The binary will be at `target/release/bedrockformation`.

**`Cargo.toml` dependencies:**

```toml
[dependencies]
md5   = "0.7"
rayon = "1"
```

---

## Usage

```
bedrockformation <seed> <x:z> <floor|roof> [x,y,z:bedrock ...]
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `seed` | `i64` | World seed |
| `x:z` | `i32:i32` | Spiral search center (your approximate coordinates) |
| `floor\|roof` | enum | Which bedrock layer to search |
| `x,y,z:bedrock` | repeatable | A block in the formation. `bedrock` is `1` (bedrock) or `0` (air). Coordinates are relative to the unknown origin. |

Blocks at Y=−64 (floor) or Y=127 (roof) are always bedrock and carry no information, so use blocks at the probabilistic layers for best results. The more blocks you provide, the rarer the formation and the faster candidates are rejected.

### Examples

**Minimal (one block, floor):**
```bash
./target/release/bedrockformation 124352345 0:0 floor 0,-63,0:1
```

**Three-block formation, floor:**
```bash
./target/release/bedrockformation 124352345 0:0 floor \
  0,-63,0:1  1,-62,0:1  0,-63,1:0
```
Block at `(0,-63,0)` should be bedrock, `(1,-62,0)` should be bedrock, `(0,-63,1)` should be air.

**Six-block formation, roof, far from origin:**
```bash
./target/release/bedrockformation 112233445 5000:5000 roof \
  0,125,0:1  1,125,0:0  0,126,0:1  -1,125,0:1  0,125,1:0  1,126,0:1
```

**Output:**
```
BedrockBlock{x=0, y=-63, z=0, shouldBeBedrock=true, p=0.800}
BedrockBlock{x=1, y=-62, z=0, shouldBeBedrock=true, p=0.600}
BedrockBlock{x=0, y=-63, z=1, shouldBeBedrock=false, p=0.800}
Found Bedrock Formation at X:-142 Z:237
```

The tool prints the blocks sorted by descending mismatch probability (most-likely-to-reject first, so it short-circuits as early as possible), then prints the result once found.

### Reading the bedrock layer

1. Stand on the bedrock floor (or build up to the roof).
2. Look straight down (or up) and record several blocks around you, noting whether each position is bedrock or air.
3. Pick any block as your relative origin `(0, y, 0)`. All other coordinates are offsets from it.
4. Use your approximate overworld coordinates as the search center `x:z`. The closer you are, the faster the search.
5. Add as many blocks as you can, since each block roughly halves the number of false positives.

---

## Benchmarks

Performance comparison between the original Java implementation and this Rust port. All times are `real` wall-clock time recorded with the shell `time` builtin. The Rust binary was compiled with `cargo build --release`. Each command was run cold (no warm JVM, no OS file cache).

**Machine:** <!-- e.g. Apple M2 Pro, 10-core / Arch Linux -->  
**Java version:** <!-- e.g. OpenJDK 21.0.3 -->  
**rustc version:** <!-- e.g. rustc 1.78.0 -->

---

### Test 1: Single loose block, floor, origin (maximum throughput)

One mid-probability block means ~50% of positions pass the filter. This measures raw candidate throughput with almost no short-circuiting.

```bash
# Java
time java -jar bedrockformation.jar 124352345 0:0 floor 0,-61,0:1

# Rust
time ./target/release/bedrockformation 124352345 0:0 floor 0,-61,0:1
```

| | `real` | `user` | `sys` |
|---|---|---|---|
| Java | | | |
| Rust | | | |

---

### Test 2: Guaranteed block, floor, origin (startup overhead)

Y=−64 is always bedrock, so every position trivially matches and the result is always the start coordinate. All measured time is process startup and initialisation.

```bash
# Java
time java -jar bedrockformation.jar 124352345 0:0 floor 0,-64,0:1

# Rust
time ./target/release/bedrockformation 124352345 0:0 floor 0,-64,0:1
```

| | `real` | `user` | `sys` |
|---|---|---|---|
| Java | | | |
| Rust | | | |

---

### Test 3: Four blocks, mixed floor probabilities, origin (realistic query)

Four blocks at different Y-levels give a combined pass rate of roughly 1-in-16. This is a typical real-world use case.

```bash
# Java
time java -jar bedrockformation.jar 124352345 0:0 floor \
  0,-63,0:1  1,-62,0:1  0,-63,1:0  -1,-61,0:1

# Rust
time ./target/release/bedrockformation 124352345 0:0 floor \
  0,-63,0:1  1,-62,0:1  0,-63,1:0  -1,-61,0:1
```

| | `real` | `user` | `sys` |
|---|---|---|---|
| Java | | | |
| Rust | | | |

---

### Test 4: Four blocks, floor, distant start (large spiral)

Same filter as test 3 but centered at X:5000 Z:5000. The match may be far from the center, exercising the spiral over many full 65 536-position chunks.

```bash
# Java
time java -jar bedrockformation.jar 987654321 5000:5000 floor \
  0,-63,0:1  1,-62,0:1  0,-63,1:0  -1,-61,0:1

# Rust
time ./target/release/bedrockformation 987654321 5000:5000 floor \
  0,-63,0:1  1,-62,0:1  0,-63,1:0  -1,-61,0:1
```

| | `real` | `user` | `sys` |
|---|---|---|---|
| Java | | | |
| Rust | | | |

---

### Test 5: Six blocks, roof, origin (tight filter)

Six roof blocks at mid-probability Y-levels give a combined pass rate of roughly 1-in-64. The most demanding filter test.

```bash
# Java
time java -jar bedrockformation.jar 112233445 0:0 roof \
  0,124,0:1  1,124,0:0  0,125,0:1  -1,124,0:1  0,124,1:0  1,125,0:1

# Rust
time ./target/release/bedrockformation 112233445 0:0 roof \
  0,124,0:1  1,124,0:0  0,125,0:1  -1,124,0:1  0,124,1:0  1,125,0:1
```

| | `real` | `user` | `sys` |
|---|---|---|---|
| Java | | | |
| Rust | | | |

---

### Test 6: Six blocks, roof, negative distant start (worst case)

Same tight filter as test 5, but centered at X:-10000 Z:-10000. Combines maximum filter difficulty with a large search space.

```bash
# Java
time java -jar bedrockformation.jar 556677889 -10000:-10000 roof \
  0,124,0:1  1,124,0:0  0,125,0:1  -1,124,0:1  0,124,1:0  1,125,0:1

# Rust
time ./target/release/bedrockformation 556677889 -10000:-10000 roof \
  0,124,0:1  1,124,0:0  0,125,0:1  -1,124,0:1  0,124,1:0  1,125,0:1
```

| | `real` | `user` | `sys` |
|---|---|---|---|
| Java | | | |
| Rust | | | |

---

### Summary

| # | Description | Java `real` | Rust `real` | Speedup |
|---|---|---|---|---|
| 1 | Single loose block, floor, origin | | | |
| 2 | Guaranteed block (startup overhead) | | | |
| 3 | Four blocks, floor, origin | | | |
| 4 | Four blocks, floor, distant start | | | |
| 5 | Six blocks, roof, origin | | | |
| 6 | Six blocks, roof, distant/negative | | | |

> **Note on JVM warmup:** all times include JVM startup (~200-400 ms depending on JRE). For long searches (tests 5-6) this is negligible; for the trivial test 2 it will dominate the Java number. This reflects real-world cold-start cost and is intentional.

---

## Credits

- [Developer-Mike](https://github.com/Developer-Mike) for the original Java implementation and RNG reverse-engineering
- Rust port, bug fixes, and performance work by <!-- your name/handle -->
