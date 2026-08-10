"""
🐉 DRAGONLENS
BDH Explainer + Microscope

DragonForge prototype:
1. 🧠 Explainer
2. 🔬 Microscope
3. 📊 Layer-by-Layer inspection

NOTE:
The current BDH model uses untrained/random weights.
Therefore numerical activation results are prototype measurements,
not results from a trained BDH checkpoint.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(CURRENT_DIR, "..", "experiments")
RESULTS_DIR = os.path.join(CURRENT_DIR, "..", "results")

sys.path.insert(0, os.path.abspath(EXPERIMENTS_DIR))

from bdh_instrumented import BDHInstrumented, BDHConfig


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DragonLens — BDH Interpretability",
    page_icon="🐉",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🐉 DragonLens")

st.subheader("BDH Interpretability & Educational Prototype")

st.markdown(
    """
    **Understand → Explore → Inspect**

    DragonLens combines a conceptual explanation of BDH with an
    interactive microscope for inspecting its internal activations.
    """
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    config = BDHConfig()

    model = BDHInstrumented(config)

    model.eval()

    return model, config


model, config = load_model()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sparsity_pct(tensor):
    """
    Percentage of values that are exactly zero.
    """

    total = tensor.numel()

    if total == 0:
        return 0.0

    zeros = (tensor == 0).sum().item()

    return round((zeros / total) * 100, 2)


def tensor_stats(tensor):

    return {
        "Shape": str(tuple(tensor.shape)),
        "Total values": tensor.numel(),
        "Active (>0)": int((tensor > 0).sum().item()),
        "Zero values": int((tensor == 0).sum().item()),
        "Sparsity": f"{sparsity_pct(tensor)}%",
        "Maximum": round(tensor.max().item(), 5),
        "Mean": round(tensor.mean().item(), 5),
    }


def run_model(text):

    """
    Convert text to byte-level tokens and run the
    instrumented BDH model.
    """

    sample_bytes = list(text.encode("utf-8"))

    # Keep values inside byte vocabulary [0,255]
    sample_bytes = [b % 256 for b in sample_bytes]

    idx = torch.tensor([sample_bytes], dtype=torch.long)

    with torch.no_grad():

        logits, loss = model(idx)

    return idx, logits


# ============================================================
# TABS
# ============================================================

explainer_tab, microscope_tab = st.tabs(
    [
        "🧠 EXPLAINER",
        "🔬 MICROSCOPE",
    ]
)


# ============================================================
# TAB 1 — EXPLAINER
# ============================================================

with explainer_tab:

    st.header("🧠 Understanding BDH")

    st.markdown(
        """
        This section gives the intuition behind the DragonForge prototype.

        The brain is used here as an **educational analogy** for selective
        and sparse computation. It should not be interpreted as claiming
        that BDH reproduces biological neural connectivity.
        """
    )

    # --------------------------------------------------------
    # BRAIN ANALOGY
    # --------------------------------------------------------

    st.subheader("1️⃣ Brain-like vs Dense Computation")

    neuron_positions = [
        (0.32, 0.37),
        (0.44, 0.26),
        (0.56, 0.29),
        (0.66, 0.39),
        (0.68, 0.58),
        (0.59, 0.71),
        (0.47, 0.76),
        (0.37, 0.68),
        (0.44, 0.50),
        (0.56, 0.53),
    ]

    sparse_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
        (0, 8),
        (8, 9),
        (9, 3),
        (8, 2),
    ]

    dense_edges = [
        (i, j)
        for i in range(len(neuron_positions))
        for j in range(i + 1, len(neuron_positions))
    ]

    network_mode = st.radio(
        "Choose a network:",
        [
            "Brain-like / sparse",
            "Dense / global",
        ],
        horizontal=True,
    )

    is_sparse = network_mode == "Brain-like / sparse"

    edges = sparse_edges if is_sparse else dense_edges

    edge_color = "#0F6E56" if is_sparse else "#D85A30"
    node_color = "#0F6E56" if is_sparse else "#D85A30"

    fig_brain = go.Figure()

    # Draw connections
    for a, b in edges:

        x0, y0 = neuron_positions[a]
        x1, y1 = neuron_positions[b]

        fig_brain.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    color=edge_color,
                    width=1.5 if is_sparse else 0.5,
                ),
                opacity=0.7 if is_sparse else 0.25,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Draw neurons
    xs = [p[0] for p in neuron_positions]
    ys = [p[1] for p in neuron_positions]

    fig_brain.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=18,
                color=node_color,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig_brain.update_layout(
        height=350,
        xaxis=dict(
            visible=False,
            range=[0.15, 0.85],
        ),
        yaxis=dict(
            visible=False,
            range=[0.15, 0.85],
            scaleanchor="x",
        ),
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_brain,
        use_container_width=True,
    )

    if is_sparse:

        st.success(
            f"""
            🧠 **{len(edges)} connections**

            The diagram illustrates the idea of selective/sparse
            connectivity as an educational analogy.
            """
        )

    else:

        st.warning(
            f"""
            🌐 **{len(edges)} connections**

            The diagram illustrates a dense connectivity pattern
            where many possible pairs are connected.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # ATTENTION EXPLANATION
    # --------------------------------------------------------

    st.subheader("2️⃣ Normal Attention vs BDH Attention")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🔵 Normal Transformer")

        st.markdown(
            """
            A standard attention mechanism typically computes
            query-key scores and then normalizes those scores.
            """
        )

        with st.expander("See simplified math"):

            st.code(
                """
scores = Q @ K.T
attention = softmax(scores)
output = attention @ V
                """,
                language="python",
            )

    with col2:

        st.markdown("### 🟢 BDH")

        st.markdown(
            """
            In the implementation used by this prototype, Q and K
            are the same tensor and the attention calculation does
            not apply a softmax.
            """
        )

        with st.expander("See BDH code idea"):

            st.code(
                """
assert K is Q

scores = QR @ KR.T
scores = scores.tril(diagonal=-1)

output = scores @ V
                """,
                language="python",
            )

    st.info(
        "💡 The attention explanation above describes the implementation "
        "used in the BDH code connected to this prototype."
    )

    st.divider()

    # --------------------------------------------------------
    # BDH PIPELINE
    # --------------------------------------------------------

    st.subheader("3️⃣ BDH Processing Pipeline")

    st.markdown(
        """
        ### Input
        ↓
        ### Embedding
        ↓
        ### Encoder
        ↓
        ### ReLU → Sparse representation
        ↓
        ### Attention
        ↓
        ### Second encoder + ReLU
        ↓
        ### Gating
        ↓
        ### Decoder
        ↓
        ### Next BDH layer
        """
    )

    st.divider()

    # --------------------------------------------------------
    # EXISTING MANIM VIDEO
    # --------------------------------------------------------

    st.subheader("🎬 BDH Explainer Animation")

    video_path = os.path.join(
        RESULTS_DIR,
        "demo",
        "BDHExplainer.mp4",
    )

    if os.path.exists(video_path):

        with open(video_path, "rb") as video_file:

            video_bytes = video_file.read()

        st.video(video_bytes)

        st.caption(
            "BDH explainer animation generated for the DragonForge prototype."
        )

    else:

        st.warning(
            "BDHExplainer.mp4 was not found in results/demo/."
        )


# ============================================================
# TAB 2 — MICROSCOPE
# ============================================================

with microscope_tab:

    st.header("🔬 BDH Microscope")

    st.markdown(
        """
        The Microscope lets you send text through the instrumented
        BDH implementation and inspect the intermediate representations
        captured from each layer.
        """
    )

    st.info(
        "⚠️ Prototype note: the current model uses untrained/random "
        "weights. The measurements below demonstrate the instrumentation "
        "and visualization pipeline rather than trained-model behavior."
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    user_text = st.text_input(
        "Enter text to inspect:",
        value="BDH is a post-transformer architecture",
    )

    inspect_button = st.button(
        "🔬 Inspect BDH",
        type="primary",
    )

    if inspect_button and user_text:

        idx, logits = run_model(user_text)

        st.success(
            f"Model executed successfully on {len(user_text)} characters."
        )

        st.divider()

        # ----------------------------------------------------
        # BASIC MODEL INFORMATION
        # ----------------------------------------------------

        st.subheader("📌 Input Information")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Characters",
            len(user_text),
        )

        c2.metric(
            "Bytes",
            idx.shape[1],
        )

        c3.metric(
            "BDH Layers",
            config.n_layer,
        )

        c4.metric(
            "Heads",
            config.n_head,
        )

        st.divider()

        # ----------------------------------------------------
        # COLLECT SPARSITY
        # ----------------------------------------------------

        sparsity_data = []

        for layer_data in model.captured_activations:

            level = layer_data["level"]

            x_sparse = layer_data["x_sparse"]
            y_sparse = layer_data["y_sparse"]
            xy_sparse = layer_data["xy_sparse"]

            sparsity_data.append(
                {
                    "Layer": level + 1,
                    "x_sparse (%)": sparsity_pct(x_sparse),
                    "y_sparse (%)": sparsity_pct(y_sparse),
                    "xy_sparse (%)": sparsity_pct(xy_sparse),
                }
            )

        df = pd.DataFrame(sparsity_data)

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader("📊 Sparsity Summary")

        avg_x = df["x_sparse (%)"].mean()
        avg_y = df["y_sparse (%)"].mean()
        avg_xy = df["xy_sparse (%)"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average x_sparse",
            f"{avg_x:.2f}%",
        )

        c2.metric(
            "Average y_sparse",
            f"{avg_y:.2f}%",
        )

        c3.metric(
            "Average gated",
            f"{avg_xy:.2f}%",
        )

        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------

        st.subheader("📈 Sparsity Across BDH Layers")

        fig = px.line(
            df,
            x="Layer",
            y=[
                "x_sparse (%)",
                "y_sparse (%)",
                "xy_sparse (%)",
            ],
            markers=True,
            title="Internal Sparsity Across Layers",
        )

        fig.update_layout(
            yaxis_title="Zero-valued activations (%)",
            xaxis_title="BDH Layer",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # ----------------------------------------------------
        # LAYER INSPECTOR
        # ----------------------------------------------------

        st.subheader("🔎 Layer Inspector")

        selected_layer = st.slider(
            "Select BDH Layer",
            min_value=1,
            max_value=config.n_layer,
            value=1,
        )

        layer_data = model.captured_activations[
            selected_layer - 1
        ]

        x_sparse = layer_data["x_sparse"]
        y_sparse = layer_data["y_sparse"]
        xy_sparse = layer_data["xy_sparse"]
        attention_output = layer_data["attention_output"]

        # ----------------------------------------------------
        # LAYER METRICS
        # ----------------------------------------------------

        st.markdown(
            f"### Layer {selected_layer}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "x_sparse sparsity",
            f"{sparsity_pct(x_sparse):.2f}%",
        )

        c2.metric(
            "y_sparse sparsity",
            f"{sparsity_pct(y_sparse):.2f}%",
        )

        c3.metric(
            "Gated sparsity",
            f"{sparsity_pct(xy_sparse):.2f}%",
        )

        # ----------------------------------------------------
        # TENSOR INFORMATION
        # ----------------------------------------------------

        st.subheader("🧮 Tensor Information")

        tensor_info = pd.DataFrame(
            [
                {
                    "Tensor": "x_sparse",
                    **tensor_stats(x_sparse),
                },
                {
                    "Tensor": "y_sparse",
                    **tensor_stats(y_sparse),
                },
                {
                    "Tensor": "xy_sparse",
                    **tensor_stats(xy_sparse),
                },
            ]
        )

        st.dataframe(
            tensor_info,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # ACTIVATION HEATMAP
        # ----------------------------------------------------

        st.subheader("🔥 Activation Heatmap")

        st.markdown(
            """
            The heatmap shows a small slice of the internal activation
            tensor so that the representation can be visually inspected.
            """
        )

        # x_sparse shape:
        # [batch, heads, tokens, N]

        x_visual = x_sparse[0]

        # Average across heads
        x_visual = x_visual.mean(dim=0)

        # Limit display to first 64 features
        x_visual = x_visual[:, :64].cpu().numpy()

        fig_heatmap = px.imshow(
            x_visual,
            aspect="auto",
            title=f"x_sparse — Layer {selected_layer}",
            labels={
                "x": "Feature",
                "y": "Token position",
                "color": "Activation",
            },
            color_continuous_scale="Blues",
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # BEFORE / AFTER RELU
        # ----------------------------------------------------

        st.subheader("⚡ Sparse Representation")

        before_relu = x_sparse.detach().flatten()

        # Show a manageable sample
        sample_size = min(
            100,
            before_relu.numel(),
        )

        sample = before_relu[:sample_size].cpu().numpy()

        activation_df = pd.DataFrame(
            {
                "Feature": range(sample_size),
                "Activation": sample,
            }
        )

        fig_activation = px.bar(
            activation_df,
            x="Feature",
            y="Activation",
            title="Sample of Sparse Activations",
        )

        st.plotly_chart(
            fig_activation,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # ATTENTION OUTPUT
        # ----------------------------------------------------

        st.subheader("🧠 Attention Output")

        attn = attention_output[0]

        # Average across attention heads
        attn_avg = attn.mean(dim=0)

        # Show first 20 features
        attn_visual = (
            attn_avg[:, :20]
            .detach()
            .cpu()
            .numpy()
        )

        fig_attn = px.imshow(
            attn_visual,
            aspect="auto",
            title=f"Attention Output — Layer {selected_layer}",
            labels={
                "x": "Internal feature",
                "y": "Token position",
                "color": "Activation",
            },
            color_continuous_scale="Viridis",
        )

        st.plotly_chart(
            fig_attn,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.success(
            f"""
            🔬 **What you are seeing**

            Layer {selected_layer} produced an internal sparse
            representation. Values equal to zero correspond to
            inactive ReLU outputs.

            The Microscope measures these values directly from the
            instrumented BDH forward pass.
            """
        )

    else:

        st.markdown(
            """
            ### 👋 Start the Microscope

            Enter some text above and click:

            **🔬 Inspect BDH**

            DragonLens will then run the text through the
            instrumented BDH model and expose its internal
            activations.
            """
        )


# ============================================================
# RESEARCH CONTRACT
# ============================================================

st.divider()

st.subheader("📜 Research Contract")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        ### 🟦 ESTABLISHED

        Claims directly supported by the BDH
        implementation used by this prototype.
        """
    )

with col2:

    st.markdown(
        """
        ### 🟩 MEASURED

        Values directly measured during a
        forward pass of the instrumented model.
        """
    )

with col3:

    st.markdown(
        """
        ### 🟨 EXPLORATORY

        Interpretations and hypotheses that
        require further investigation.
        """
    )

st.caption(
    "🐉 DragonLens — DragonForge prototype | "
    "BDH Interpretability & Educational Visualization"
)