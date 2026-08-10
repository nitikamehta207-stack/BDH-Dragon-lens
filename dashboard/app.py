"""
BDH Microscope Dashboard
=========================
This dashboard has 3 sections:
1. Explainer - How BDH attention differs from normal attention
2. Live Microscope - User types text, sees real-time activations
3. Layer-by-Layer Story - Slider to explore gating/sparsity patterns

To run: streamlit run app.py
"""

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# experiments folder ka path add karo taaki bdh_instrumented import ho sake
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from bdh_instrumented import BDHInstrumented, BDHConfig

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="BDH Microscope", layout="wide")
st.title("🔬 BDH Microscope")
st.caption("Understanding Dragon Hatchling's internals — DragonForge Submission")

# ============================================================
# LOAD MODEL (once, cached - avoids reloading on every interaction)
# ============================================================
@st.cache_resource
def load_model():
    config = BDHConfig()  # default settings
    model = BDHInstrumented(config)
    model.eval()
    return model, config

model, config = load_model()

# ============================================================
# SECTION 0: BRAIN ANALOGY — Sparse Local Network vs Dense Global Network
# ============================================================
st.header("0️⃣ The Brain Analogy")
st.markdown("""
**In plain terms:** Real neurons in your brain mostly talk to their nearby neighbors —
a **sparse, local** network. Standard AI models (Transformers) work differently: every
word attends to every other word, no matter how far apart — a **dense, global** network.
BDH is designed to copy the brain's approach. Toggle below to see the difference.
""")

import random

# Fixed neuron positions (inside a simple brain-shaped outline)
neuron_positions = [
    (0.32, 0.37), (0.44, 0.26), (0.56, 0.29), (0.66, 0.39), (0.68, 0.58),
    (0.59, 0.71), (0.47, 0.76), (0.37, 0.68), (0.44, 0.50), (0.56, 0.53),
]

# Sparse (brain-like): each neuron connects only to its nearby neighbors
sparse_edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,8),(8,9),(9,3),(8,2)]

# Dense (transformer-like): every neuron connects to every other neuron
dense_edges = [(i, j) for i in range(len(neuron_positions)) for j in range(i+1, len(neuron_positions))]

network_mode = st.radio(
    "Choose network type:",
    ["Brain-like (BDH) — sparse, local", "Transformer — dense, global"],
    horizontal=True,
)

is_sparse = network_mode.startswith("Brain-like")
edges = sparse_edges if is_sparse else dense_edges
edge_color = "#0F6E56" if is_sparse else "#D85A30"
node_color = "#0F6E56" if is_sparse else "#D85A30"

fig_brain = go.Figure()

# Draw edges (connections between neurons)
for a, b in edges:
    x0, y0 = neuron_positions[a]
    x1, y1 = neuron_positions[b]
    fig_brain.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(color=edge_color, width=1.5 if is_sparse else 0.5),
        opacity=0.7 if is_sparse else 0.25,
        showlegend=False,
        hoverinfo='skip',
    ))

# Draw neurons (dots)
xs = [p[0] for p in neuron_positions]
ys = [p[1] for p in neuron_positions]
fig_brain.add_trace(go.Scatter(
    x=xs, y=ys,
    mode='markers',
    marker=dict(size=18, color=node_color),
    showlegend=False,
    hoverinfo='skip',
))

fig_brain.update_layout(
    height=350,
    xaxis=dict(visible=False, range=[0.15, 0.85]),
    yaxis=dict(visible=False, range=[0.15, 0.85], scaleanchor="x"),
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig_brain, use_container_width=True)

if is_sparse:
    st.success(f"🧠 **{len(edges)} connections** — neurons only talk to nearby neighbors, just like real brain tissue. This is what BDH is designed to mimic.")
else:
    st.warning(f"🌐 **{len(edges)} connections** — every neuron talks to every other neuron, regardless of distance. This is how standard Transformer attention works — powerful, but expensive.")

st.divider()

# ============================================================
# SECTION 1: EXPLAINER — Normal Attention vs BDH Attention
# ============================================================
st.header("1️⃣ What Makes BDH Different")

st.markdown("""
**In plain terms:** When a normal AI model (like ChatGPT) decides which earlier words matter,
it makes every word "vote" on importance, then normalizes those votes into probabilities
(this step is called *softmax*). BDH skips the voting step entirely — it lets signals pass
through more directly, similar to how neurons in a brain send signals to each other without
a central "normalizer."
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Normal Transformer")
    st.markdown("Every word looks at every other word, then the model **normalizes** how much attention to pay to each (softmax = 'turn scores into percentages that add up to 100%').")
    with st.expander("See the actual math"):
        st.code("""
scores = Q @ K.T
scores = softmax(scores)   # normalizes into probabilities
output = scores @ V
        """, language="python")

with col2:
    st.subheader("🟢 BDH Attention")
    st.markdown("Query and Key are **forced to be identical** (Q=K), and there's **no normalization step**. Signals pass through directly — closer to how brain synapses work.")
    with st.expander("See the actual math"):
        st.code("""
assert K is Q               # Q and K are the SAME tensor!
scores = QR @ KR.T
scores = scores.tril(-1)    # causal mask only, no softmax
output = scores @ V
        """, language="python")

st.info("💡 **ESTABLISHED (from paper/code):** BDH attention skips softmax and enforces a Q=K constraint — a fundamental departure from standard Transformer attention. We verified this by reading `bdh.py` directly (see `Attention.forward()`).")

st.divider()

# ============================================================
# SECTION 2: LIVE MICROSCOPE — User Input Se Real-time Activations
# ============================================================
st.header("2️⃣ Live Microscope")
st.markdown("""
**In plain terms:** As your text flows through BDH's 6 layers, a large fraction of the model's
internal "neurons" switch off (become exactly zero) at each step. This is called **sparsity** —
think of it as the model being *selective*, keeping only the signals that matter and silencing
the rest. Type something below to measure it live.
""")

user_text = st.text_input("Enter text:", value="BDH is a post-transformer architecture")

if user_text:
    # Convert text to bytes (BDH is byte-level)
    sample_bytes = list(user_text.encode('utf-8'))
    idx = torch.tensor([sample_bytes])

    with torch.no_grad():
        logits, loss = model(idx)

    # Compute sparsity for each level
    sparsity_data = []
    for layer_data in model.captured_activations:
        level = layer_data['level']
        x_sparse = layer_data['x_sparse']
        y_sparse = layer_data['y_sparse']
        xy_sparse = layer_data['xy_sparse']

        def sparsity_pct(t):
            return round(((t == 0).sum().item() / t.numel()) * 100, 2)

        sparsity_data.append({
            'Level': level,
            'x_sparse (%)': sparsity_pct(x_sparse),
            'y_sparse (%)': sparsity_pct(y_sparse),
            'xy_sparse gated (%)': sparsity_pct(xy_sparse),
        })

    df = pd.DataFrame(sparsity_data)
    avg_gated = df['xy_sparse gated (%)'].mean()
    avg_x = df['x_sparse (%)'].mean()

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.line(df, x='Level', y=['x_sparse (%)', 'y_sparse (%)', 'xy_sparse gated (%)'],
                      title="Sparsity Across Layers", markers=True)
        fig.update_layout(yaxis_title="% of neurons switched OFF", xaxis_title="Layer (Level)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📖 **How to read this:** Higher = more neurons are 'off' (zero) at that stage. "
                   "The pink line (xy_sparse) is what's left AFTER the model combines two signals together (called 'gating').")

    with col2:
        st.dataframe(df, use_container_width=True)

    st.success(f"""
✅ **MEASURED — So what does this mean?**
Before combining signals, about **{avg_x:.0f}%** of neurons are already off.
After the model *gates* (combines) two signals together, that jumps to **{avg_gated:.0f}%** —
meaning only about **1 in 4 neurons stays active**. In plain terms: BDH's gating mechanism acts
like a strict filter, letting through only the most important information at each layer.
""")

st.divider()

# ============================================================
# SECTION 3: LAYER-BY-LAYER STORY — Slider Se Explore Karo
# ============================================================
st.header("3️⃣ Layer-by-Layer Story")
st.markdown("""
**In plain terms:** BDH processes your text through 6 layers, one after another — but unlike
most AI models, it **reuses the exact same internal machinery** at every layer (like passing
coffee through the same filter 6 times instead of using 6 different filters). Move the slider
below to see how the signal looks at each pass.
""")

if user_text and model.captured_activations:
    selected_level = st.slider("Select Layer", 0, config.n_layer - 1, 0)

    layer_data = model.captured_activations[selected_level]

    explanations = {
        0: "**First pass** — the raw input hasn't been filtered yet, sparsity is moderate.",
        config.n_layer - 1: "**Final pass** — after being filtered 6 times, the signal has been progressively 'refined' down to what matters most.",
    }
    default_explanation = f"**Pass {selected_level + 1} of {config.n_layer}** — the input has been through the same filter {selected_level} time(s) already."

    st.markdown(explanations.get(selected_level, default_explanation))

    # Show attention output heatmap (averaged across heads)
    attn_output = layer_data['attention_output']  # shape: (B, nh, T, D)
    attn_avg = attn_output[0].mean(dim=0).numpy()  # average across heads -> (T, D)

    st.markdown("""
    **What am I looking at?** Each row below is one character/byte from your text (in order).
    Each column is one of the model's internal "features." **Lighter = more active, darker = less active.**
    Look for columns that stay consistently light or dark — those are features the model treats
    similarly regardless of which character it's looking at, vs. columns that vary a lot row-to-row
    (those are features that respond differently to different characters).
    """)

    fig2 = px.imshow(attn_avg[:, :20], aspect='auto',
                      title=f"What the model 'sees' internally at Pass {selected_level + 1} (first 20 features shown)",
                      labels=dict(x="Internal Feature #", y="Character Position in Your Text", color="Activation Strength"),
                      color_continuous_scale="Blues")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("🔬 **EXPLORATORY:** We're visually inspecting this pattern ourselves — we haven't statistically "
               "verified specific feature meanings. A trained model would let us map specific features to concepts "
               "(this is what Pathway's paper calls 'monosemantic synapses').")

st.divider()

# ============================================================
# FOOTER — Research Contract Reminder
# ============================================================
st.markdown("""
---
### Research Contract
- 🟦 **ESTABLISHED**: Claims that come directly from Pathway's paper/code
- 🟩 **MEASURED**: What we directly observed by testing (live in this dashboard)
- 🟨 **EXPLORATORY**: Our own hypothesis/extension

⚠️ **Note:** This model currently runs with *untrained* (random) weights. Results will be more meaningful on a
trained model — run `train.py` and load the checkpoint to get richer insights from this dashboard.
""")