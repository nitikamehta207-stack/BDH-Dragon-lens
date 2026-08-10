from manim import *


class BDHExplainer(Scene):
    def construct(self):
        self.intro()
        self.brain_basics()
        self.sparse_vs_dense()
        self.bdh_gating()
        self.outro()

    # ------------------------------------------------------------
    # SCENE 1: Title / Intro
    # ------------------------------------------------------------
    def intro(self):
        title = Text("BDH: Dragon Hatchling", font_size=48, color=BLUE)
        subtitle = Text("Bridging Brains and Transformers", font_size=28, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

    # ------------------------------------------------------------
    # SCENE 2: How Real Neurons Work
    # ------------------------------------------------------------
    def brain_basics(self):
        heading = Text("How Real Neurons Work", font_size=36).to_edge(UP)
        self.play(Write(heading))

        positions = [
            LEFT * 3 + UP * 1, LEFT * 1.5 + UP * 1.8, RIGHT * 0.5 + UP * 1.5,
            RIGHT * 2.5 + UP * 0.8, RIGHT * 3 + DOWN * 0.5, RIGHT * 1.5 + DOWN * 1.8,
            LEFT * 0.5 + DOWN * 2, LEFT * 2.5 + DOWN * 1, LEFT * 0.8 + UP * 0.2,
            RIGHT * 1 + DOWN * 0.3,
        ]
        neurons = VGroup(*[Dot(point=p, radius=0.13, color=BLUE_C) for p in positions])

        sparse_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                         (6, 7), (7, 0), (0, 8), (8, 9), (9, 3), (8, 2)]
        edges = VGroup(*[
            Line(neurons[a].get_center(), neurons[b].get_center(),
                 stroke_width=1.5, color=GRAY_C, stroke_opacity=0.6)
            for a, b in sparse_pairs
        ])

        self.play(LaggedStart(*[Create(n) for n in neurons], lag_ratio=0.1))
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.05))
        self.wait(0.5)

        explanation = Text(
            "Each neuron mostly talks to its nearby neighbors",
            font_size=24, color=GRAY_A
        ).to_edge(DOWN)
        self.play(Write(explanation))

        signal_path = [0, 1, 2, 3, 4]
        for i in range(len(signal_path) - 1):
            a, b = signal_path[i], signal_path[i + 1]
            self.play(neurons[a].animate.set_color(YELLOW), run_time=0.25)
            self.play(neurons[b].animate.set_color(YELLOW), run_time=0.25)
        self.wait(1)

        self.play(
            FadeOut(heading), FadeOut(neurons), FadeOut(edges), FadeOut(explanation)
        )

    # ------------------------------------------------------------
    # SCENE 3: Sparse (Brain) vs Dense (Transformer)
    # ------------------------------------------------------------
    def sparse_vs_dense(self):
        heading = Text("Brain-like (Sparse) vs Transformer (Dense)", font_size=32).to_edge(UP)
        self.play(Write(heading))

        left_positions = [
            LEFT * 5 + UP * 1.2, LEFT * 3.8 + UP * 1.8, LEFT * 2.8 + UP * 1,
            LEFT * 2.5 + DOWN * 0.5, LEFT * 3.5 + DOWN * 1.5, LEFT * 5 + DOWN * 1,
        ]
        left_neurons = VGroup(*[Dot(p, radius=0.11, color=GREEN_C) for p in left_positions])
        left_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        left_edges = VGroup(*[
            Line(left_neurons[a].get_center(), left_neurons[b].get_center(),
                 stroke_width=1.5, color=GREEN_C, stroke_opacity=0.7)
            for a, b in left_pairs
        ])
        left_label = Text("Sparse: 6 connections", font_size=20, color=GREEN_C)
        left_label.next_to(VGroup(*left_neurons), DOWN, buff=0.5)

        right_positions = [
            RIGHT * 2.5 + UP * 1.2, RIGHT * 3.7 + UP * 1.8, RIGHT * 4.7 + UP * 1,
            RIGHT * 5 + DOWN * 0.5, RIGHT * 4 + DOWN * 1.5, RIGHT * 2.5 + DOWN * 1,
        ]
        right_neurons = VGroup(*[Dot(p, radius=0.11, color=RED_C) for p in right_positions])
        right_pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        right_edges = VGroup(*[
            Line(right_neurons[a].get_center(), right_neurons[b].get_center(),
                 stroke_width=0.8, color=RED_C, stroke_opacity=0.3)
            for a, b in right_pairs
        ])
        right_label = Text("Dense: 15 connections", font_size=20, color=RED_C)
        right_label.next_to(VGroup(*right_neurons), DOWN, buff=0.5)

        self.play(Create(left_neurons), Create(right_neurons))
        self.play(Create(left_edges), Create(right_edges))
        self.play(Write(left_label), Write(right_label))
        self.wait(2)

        conclusion = Text(
            "BDH is designed to work like the brain: sparse, not dense",
            font_size=24, color=YELLOW
        ).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(1.5)

        self.play(
            FadeOut(heading), FadeOut(left_neurons), FadeOut(right_neurons),
            FadeOut(left_edges), FadeOut(right_edges),
            FadeOut(left_label), FadeOut(right_label), FadeOut(conclusion),
        )

    # ------------------------------------------------------------
    # SCENE 4: BDH's Gating Mechanism (Sparsity Funnel)
    # ------------------------------------------------------------
    def bdh_gating(self):
        heading = Text("BDH's Gating: Filtering Signals", font_size=32).to_edge(UP)
        self.play(Write(heading))

        stages = ["x_sparse", "y_sparse", "gated (x \u00d7 y)"]
        values = [50, 51, 75]
        colors = [BLUE_C, TEAL_C, PURPLE_C]

        bars = VGroup()
        labels = VGroup()
        percent_labels = VGroup()

        bar_width = 1.2
        max_height = 4

        for i, (stage, val, color) in enumerate(zip(stages, values, colors)):
            height = (val / 100) * max_height
            bar = Rectangle(width=bar_width, height=height, color=color, fill_opacity=0.8)
            bar.move_to(RIGHT * (i - 1) * 2.2 + DOWN * (max_height / 2 - height / 2) + UP * 0.5)

            label = Text(stage, font_size=20).next_to(bar, DOWN, buff=0.3)
            pct_label = Text(f"{val}%", font_size=24, color=color).next_to(bar, UP, buff=0.2)

            bars.add(bar)
            labels.add(label)
            percent_labels.add(pct_label)

        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.3))
        self.play(Write(labels), Write(percent_labels))
        self.wait(1)

        explanation = Text(
            "Combining two sparse signals (gating) filters out even more —\n"
            "only the most important information survives.",
            font_size=22, color=GRAY_A
        ).to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2.5)

        self.play(
            FadeOut(heading), FadeOut(bars), FadeOut(labels),
            FadeOut(percent_labels), FadeOut(explanation),
        )

    # ------------------------------------------------------------
    # SCENE 5: Conclusion
    # ------------------------------------------------------------
    def outro(self):
        summary1 = Text("BDH combines brain-like sparsity", font_size=32, color=BLUE)
        summary2 = Text("with Transformer-level performance", font_size=32, color=BLUE)
        summary2.next_to(summary1, DOWN, buff=0.3)

        self.play(Write(summary1))
        self.play(Write(summary2))
        self.wait(1)

        footer = Text("DragonForge Submission — BDH Microscope", font_size=20, color=GRAY_B)
        footer.next_to(summary2, DOWN, buff=0.8)
        self.play(FadeIn(footer))
        self.wait(2)

        self.play(FadeOut(summary1), FadeOut(summary2), FadeOut(footer))