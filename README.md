<div align="center">

# Minecraft bedrock finder

###### Find any bedrock pattern blazingly fast.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/e7adb365-d5f3-44b5-b516-2eaff2739b69"/>

</div>

<br>

This tool searches a world's seed spiraling outwards from a given coordinate _(0,0 by default)_, until it finds the pattern specified.

---

> TODO: table of contents

---

## Installation

### Linux

> [!NOTE]
> To build this project, you must have Rust and Cargo installed.

```bash
# Firstly, clone the repo.
git clone https://github.com/akai-hana/minecraft-bedrock-finder

# Go into the cloned directory...
cd minecraft-bedrock-finder

# ... And finally compile the project.
cargo build --release
```

And that's it!

Now you can then execute it by doing:

```bash
./target/release/bedrockformation
```

..., where the binary is located.

<sub>*(or just going there and double clicking the file)*</sub>

## Lore

This project started as a Rust port of [Developer-Mike's Minecraft Bedrock Formation Finder](https://github.com/Developer-Mike/Minecraft-Bedrock-Formation-Finder-1.18).

Now, it could be considered entirely its own thing, featuring many new, very juicy features.

## Features

### GUI

Most prominently, this program offers an intuitive GUI to search the bedrock patterns.

(Photo (im too lazy ill do this later))

The GUI is written using Iced, a library to make GUIs in Rust (the same library the guys at System76 use to write their COSMIC desktop)

### GPU Acceleration

This program offers two modes of computing the pattern searches: though CPU or GPU. 

GPU acceleration speeds the searches nearly tenfold as compared to CPU mode _(depending on your GPU, but mostly the same),_ so it is advised to check the checkbox on the interface.

Regardless, the CPU mode this program offers the is already way, WAY faster and plenty more optimized in comparison to the competition, completely decimating it in the process *(concrete benchmarks below if interested).*

As an example, Developer-Mike's (originally this program's fork) version works really well, but since it was written in Java, and the search computation logic relies on OOP, the searches are inherently slow. In comparison, this program is written in Rust, and offers multi-threading, parallelism, and GPU acceleration, making the search absurdly fast.

(Benchmarks of stuff (im still lazy))

---

## Benchmarks

> [!WARNING]
> These benchmarks are outdated.
> Once they are updated, you won't see this warning anymore. Until then, please wait a bit.

Performance comparison between the original Java implementation and this Rust port. All times are `real` wall-clock time recorded with the shell `time` builtin. The Rust binary was compiled with `cargo build --release`. Each command was run cold (no warm JVM, no OS file cache).

**CPU: Ryzen 9 6900HX**
**GPU: RX 6850M XT**
**RAM: 32GB DDR4**
**Java version: <TODO>**
**rustc version: <TODO>**

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
- Rust port, bug fixes, performance work, GUI, GPU acceleration, and everything else in-between by [akai-hana](https://github.com/akai-hana) <sub>*AKA. me :-)*</sub> and commisioned by [More$!@#%](https://github.com/MoreOrgasm) <sub>sorry, not allowed to spell that one :-(</sub>https://github.com/MoreOrgasm
