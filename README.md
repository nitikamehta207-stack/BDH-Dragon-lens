# BDH Microscope — Understanding Dragon Hatchling's Internals

## One Sentence Claim

An interactive dashboard and animated explainer that measures BDH's sparse, brain-inspired gating mechanism in real time, verifying that multiplicative gating pushes sparsity from ~50% to ~75% — consistent with Pathway's published claims about brain-like sparse activity.

## Problem / Research Question

BDH (Dragon Hatchling) is a newly released, brain-inspired alternative to the Transformer architecture. Because it is so new, there are no accessible tools that let a non-expert see *what's actually happening inside* the model, or connect its behavior back to the neuroscience principles it claims to be inspired by.

**Our question:** Can we build a tool that (1) measures BDH's internal sparsity and gating behavior directly from the public source code, and (2) explains *why it matters* — in brain-science terms — to someone without a deep learning background?

## Architecture

```
User Input (text)
      │
      ▼
BDHInstrumented (modified bdh.py — captures activations at every layer)
      │
      ├──► analyze_sparsity.py ──► results/sparsity_results.csv
      │
      └──► Streamlit Dashboard (dashboard/app.py)
              ├─ Section 0: Brain-like (sparse) vs Transformer (dense) network comparison
              ├─ Section 1: BDH attention vs normal attention (no softmax, Q=K constraint)
              ├─ Section 2: Live sparsity measurement (user types text, sees real-time graph)
              └─ Section 3: Layer-by-layer attention pattern explorer

Manim Animation (manim_animation / dashboard/animations)
      └─ 5-scene explainer video connecting brain neuroscience to BDH's gating mechanism
```

## How to Run

```bash
# 1. Clone this repo and enter it
git clone https://github.com/nitikamehta207-stack/BDH-Dragon-lens.git
cd BDH-Dragon-lens

# 2. Install dependencies
pip install -r requirements.txt
pip install streamlit plotly pandas

# 3. Run the sparsity analysis (verifies instrumentation works)
cd experiments
python analyze_sparsity.py

# 4. Launch the interactive dashboard
cd ../dashboard
python -m streamlit run app.py
```

The dashboard opens automatically in your browser at `http://localhost:8501`.

**To reproduce the animation** (requires Manim — recommended via Google Colab, see `manim_animation/bdh_explainer.py` header for full instructions):
```python
%manim -qh BDHExplainer
```

## Proof

- **Live sparsity measurements:** `results/sparsity_results.csv` — sparsity % for every layer (0–5), for both individual signals (`x_sparse`, `y_sparse`) and their gated product (`xy_sparse`)
- **Interactive dashboard:** `dashboard/app.py` — lets anyone type their own text and see the same measurement live
- **Animated explainer:** `results/demo/BDHExplainer.mp4` — 5-scene video connecting brain science to BDH's design
- **Screenshots:** `results/` (dashboard screenshots)

### Key Measured Result

| Stage | Average Sparsity |
|---|---|
| `x_sparse` (after first ReLU) | ~50% |
| `y_sparse` (after attention + second ReLU) | ~51% |
| `xy_sparse` (after gating — multiplying the two) | **~75%** |

This pattern was consistent across all 6 layers and multiple input texts.

## Research Labels

- 🟦 **ESTABLISHED** (from Pathway's paper/code, verified by reading `bdh.py` directly):
  - BDH attention has no softmax normalization step and enforces a Q=K constraint (`assert K is Q` in `Attention.forward()`), unlike standard Transformer attention.
  - The same encoder/attention/decoder weights are reused across all 6 layers (`for level in range(C.n_layer)` loop reuses `self.encoder`, `self.attn`, `self.decoder` every iteration) — matching the paper's "shared parameters across layers" claim.
  - BDH is explicitly designed around brain-inspired principles: local interaction, sparse activity, and Hebbian-style memory (per the BDH README and explainer).

- 🟩 **MEASURED** (observed directly by us, via `bdh_instrumented.py`):
  - On an untrained (random-weight) model, gating (`x_sparse * y_sparse`) produces ~75% sparsity — significantly higher than either individual component (~50%). This pattern held consistently across all 6 layers and several different input texts.

- 🟨 **EXPLORATORY** (our hypothesis, not yet statistically verified):
  - We hypothesize that this gating-amplified sparsity effect would be even more pronounced in a trained model, where the paper's "predictability-linked activity" claim (activity gets quieter as input becomes more predictable) should become measurable. We were not able to fully train the model within the hackathon timeframe to test this directly.

## Limitations

- All measurements in this submission were taken on an **untrained (random-weight) model** due to hackathon time constraints. The paper's core claims (monosemantic synapses, predictability-linked activity) are best observed in substantially trained models — our results confirm the *architectural* sparsity behavior (which is present even untrained, since it comes from ReLU + gating structure), but do not yet confirm the *learned* interpretability claims.
- The brain-like vs dense-Transformer network visualization (Section 0 of the dashboard) is an **illustrative diagram**, not a live measurement — the connection counts (12 vs 45) are chosen for teaching clarity, not derived from BDH's actual internal connectivity graph.
- We tested on short English text samples only; we have not evaluated multilingual or long-context behavior.

## Team Contributions

1. Nitika-Explainer Model implementation using Manim library
2. Shubhangi-Testing and presentation
3. Anushka- Documentation experiments and Dashboard(Microscope Tool)
4. Ayushi- Research and analysis

## If We Had Access to a Larger BDH Model

We would run the same `bdh_instrumented.py` pipeline against a substantially trained, larger BDH checkpoint and compare:
- **Metric:** Layer-wise sparsity (`x_sparse`, `y_sparse`, `xy_sparse` percentages), exactly as measured in this repo
- **Comparison:** Untrained (this submission) vs trained, and small vs large parameter count
- **Hypothesis to test:** That sparsity becomes more *input-dependent* in trained/larger models (i.e., more predictable inputs produce measurably quieter activity, per Pathway's "predictability-linked activity" claim) — something that is architecturally impossible to observe in an untrained model regardless of size.

## Technology / Research Anchor

- **BDH source:** `bdh.py`, cloned from Pathway's public repository (github.com/pathwaycom/bdh), used unmodified in `bdh_core/`
- **Instrumentation:** `experiments/bdh_instrumented.py` — a copy of `bdh.py` with activation-capture added inside the forward loop (no hooks needed, since the sparse tensors are local variables, not module outputs)
- **Dashboard:** Streamlit + Plotly
- **Animation:** Manim Community Edition (rendered via Google Colab)
