1. Bug: Multi-rotation result priority — Ok(None) can be overridden by Err
In the multi-rotation result collection loop inside Message::Search:
rustlet mut last: Result<Option<(i32, i32)>, String> = Ok(None);
for r in results {
    match r {
        Ok(Some(_)) => return r,
        Ok(None)    => { last = Ok(None); }
        Err(e)      => {
            if matches!(last, Ok(None)) {  // ← triggers on *initial* Ok(None) too
                last = Err(e);
            }
        }
    }
}
The comment says "Only downgrade to an error if we have no cancellation to report yet", but last is initialised to Ok(None) — the same value a cancelled rotation would return. So if the sequence is [Ok(None), Err(e)] (rotation 0 cancelled, rotation 1 has an impossible pattern), the Err incorrectly overwrites the cancellation. Suggested fix:
rustlet mut last: Result<Option<(i32, i32)>, String> = Ok(None);
let mut saw_cancellation = false;
for r in results {
    match r {
        Ok(Some(_)) => return r,
        Ok(None)    => { saw_cancellation = true; last = Ok(None); }
        Err(e)      => {
            if !saw_cancellation { last = Err(e); }
        }
    }
}
In practice this is unreachable today — generate_rotations only changes X/Z offsets, so impossibility (which depends only on Y) is shared by all rotations or none. But the code is wrong and the comment is misleading, so it's worth fixing before a future refactor triggers it.

2. BedrockType::min() / ::max() naming — confusing for Roof
rustfn min(self) -> i32 { match self { BedrockType::Floor => -64, BedrockType::Roof => 128 } }
fn max(self) -> i32 { match self { BedrockType::Floor => -59, BedrockType::Roof => 123 } }
For Roof, min() returns 128 and max() returns 123 — so min() > max(). This is jarring to any reader. The semantics are really "always-solid Y" and "always-air boundary Y". A doc comment on the methods explaining the Roof inversion would go a long way:
rust/// The Y coordinate at which this bedrock layer is always solid (probability ≥ 1).
/// For Floor this is the bottom (-64); for Roof this is the top (128).
fn always_solid_y(self) -> i32 { ... }

/// The Y coordinate at which this bedrock layer is always air (probability ≤ 0).
/// For Floor this is -59; for Roof this is 123.
fn always_air_y(self) -> i32 { ... }
Or, if you keep min/max, at least add a comment like // NOTE: for Roof, min > max; see compute_probability.

3. Rotation formula comment has stray * characters
In rotate_blocks:
rust/// Rotation formulae (standard 2-D, with X east and Z south):
///   0º -> (x,  z)
///   1* CW  ->  (−z,  x)   // ← should be 1× or 90°
///   2* CW  ->  (−x, −z)
///   3* CW  ->  ( z, −x)
1* / 2* / 3* look like formatting artifacts. Change to 1×, 2×, 3× or 90° CW, 180° CW, 270° CW.

4. zoom_row doesn't use sc() for its internal spacing
Most of the view() code consistently uses sc(v) for all pixel measurements so things scale with zoom. But zoom_row has two hardcoded values:
rustSpace::with_width(Length::Fixed(8.0)),   // not sc(8.0)
].spacing(4).align_items(...)            // not sc(4.0) as u16
The zoom controls' internal spacing won't scale when the user zooms in/out, making them look cramped at high zoom. Change to sc(8.0) and sc(4.0) as u16.

5. Dead commented-out code in view()
rust// .push(horizontal_rule(1))
// .push(Space::with_height(Length::Fixed(12.0)))
These two lines in the content column appear to have been experimentally removed and then left. Safe to delete.

6. compute_probability called twice for the same block
In Message::Search:
rustblocks_vec.push(Block {
    probability:    compute_probability(y, bt),
    prob_threshold: prob_to_threshold(compute_probability(y, bt)),  // called again
    ...
});
Minor, but easy to fix:
rustlet prob = compute_probability(y, bt);
blocks_vec.push(Block {
    probability:    prob,
    prob_threshold: prob_to_threshold(prob),
    ...
});

7. SEARCH_BATCH_SIZE comment says "AVX-512 group size"
rustconst SEARCH_BATCH_SIZE: i64 = 1 << 20; // ... must be a multiple of 8 (AVX-512 group size)
Groups are always 8 positions wide — for the scalar path, AVX2 path, and AVX-512 path alike. "AVX-512 group size" implies the constraint is specific to AVX-512, which it isn't. Change to just "SIMD group size" or "positions per group".
