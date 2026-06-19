from manim import *
import numpy as np
import math


S0 = 100.0
MU = 0.10
SIGMA = 0.25
T_MAX = 1.0
N_STEPS = 250

K = 100.0
R = 0.05

SEED = 42


def gbm_path(s0, mu, sigma, T, n_steps, rng):
    dt = T / n_steps
    z = rng.standard_normal(n_steps)
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_path = np.log(s0) + np.cumsum(log_increments)
    return np.concatenate([[s0], np.exp(log_path)])


def gbm_terminal_samples(s0, mu, sigma, T, n_samples, rng):
    z = rng.standard_normal(n_samples)
    return s0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(s0, k, r, sigma, T):
    if T <= 0:
        return max(s0 - k, 0.0)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * norm_cdf(d1) - k * math.exp(-r * T) * norm_cdf(d2)


class BlackScholesJourney(Scene):
    """One continuous narrative: noise -> GBM -> distribution -> Ito -> PDE -> price."""

    def construct(self):
        self.act_1_brownian_walk()
        self.act_2_gbm_fan_out()
        self.act_3_terminal_distribution()
        self.act_4_ito_lemma()
        self.act_5_pde()
        self.act_6_price_and_convergence()


    # ACT 1 — a single noisy path. Build intuition for dW.
  
    def act_1_brownian_walk(self):
        title = Text("A particle that can't decide where to go", font_size=36).to_edge(UP)
        self.play(Write(title))

        rng = np.random.default_rng(SEED)
        n = 200
        dt = T_MAX / n
        increments = rng.standard_normal(n) * math.sqrt(dt)
        w = np.concatenate([[0.0], np.cumsum(increments)])
        t_vals = np.linspace(0, T_MAX, n + 1)

        axes = Axes(
            x_range=[0, T_MAX, 0.2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=5,
            axis_config={"include_numbers": True},
        ).shift(DOWN * 0.3)
        x_label = axes.get_x_axis_label("t")
        y_label = axes.get_y_axis_label("W_t")
        self.play(Create(axes), Write(x_label), Write(y_label))

        time_tracker = ValueTracker(0.0)

        def current_index():
            frac = time_tracker.get_value() / T_MAX
            return max(1, int(frac * n))

        path = always_redraw(
            lambda: axes.plot_line_graph(
                t_vals[: current_index() + 1],
                w[: current_index() + 1],
                line_color=BLUE,
                add_vertex_dots=False,
                stroke_width=3,
            )
        )
        dot = always_redraw(
            lambda: Dot(axes.c2p(t_vals[current_index()], w[current_index()]), color=YELLOW, radius=0.07)
        )

        eq = MathTex(r"dW_t \sim \mathcal{N}(0, \, dt)").to_corner(UR).shift(DOWN * 0.5)

        self.add(path, dot)
        self.play(Write(eq))
        self.play(time_tracker.animate.set_value(T_MAX), run_time=5, rate_func=linear)

        note = Text("Independent increments. No memory. Pure noise.", font_size=28).to_edge(DOWN)
        self.play(Write(note))
        self.wait(1.5)

        self.play(
            FadeOut(title), FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
            FadeOut(eq), FadeOut(note), FadeOut(path), FadeOut(dot),
        )

  
    # ACT 2 — GBM paths fan out.
   
    def act_2_gbm_fan_out(self):
        title = Text("Now give the noise a drift and scale it by price", font_size=32).to_edge(UP)
        eq = MathTex(r"dS_t = \mu S_t\, dt + \sigma S_t\, dW_t").next_to(title, DOWN)
        self.play(Write(title), Write(eq))

        rng = np.random.default_rng(SEED)
        n_paths = 40
        t_vals = np.linspace(0, T_MAX, N_STEPS + 1)
        paths = [gbm_path(S0, MU, SIGMA, T_MAX, N_STEPS, rng) for _ in range(n_paths)]

        axes = Axes(
            x_range=[0, T_MAX, 0.25],
            y_range=[40, 220, 20],
            x_length=10.5,
            y_length=5,
            axis_config={"include_numbers": True},
        ).shift(DOWN * 0.5)
        self.play(Create(axes), eq.animate.next_to(axes, UP, buff=0.15))

        time_tracker = ValueTracker(0.0)

        def idx():
            frac = time_tracker.get_value() / T_MAX
            return max(1, int(frac * N_STEPS))

        def make_path_mobject(path_array, color):
            return always_redraw(
                lambda: axes.plot_line_graph(
                    t_vals[: idx() + 1],
                    path_array[: idx() + 1],
                    line_color=color,
                    add_vertex_dots=False,
                    stroke_width=2,
                    stroke_opacity=0.55,
                )
            )

        path_mobjects = []
        for p in paths:
            color = GREEN if p[-1] > S0 else (RED if p[-1] < S0 * 0.9 else GREY_B)
            path_mobjects.append(make_path_mobject(p, color))

        start_dot = Dot(axes.c2p(0, S0), color=YELLOW)
        s0_label = MathTex("S_0").next_to(start_dot, LEFT, buff=1)

        self.add(*path_mobjects)
        self.play(FadeIn(start_dot), Write(s0_label))
        self.play(time_tracker.animate.set_value(T_MAX), run_time=6, rate_func=linear)

        note = Text(
            "Same equation, every path. Different noise, wildly different futures.", font_size=18
            
        ).to_edge(RIGHT)
        self.play(Write(note))
        self.wait(1.5)

        self.play(
            FadeOut(title), FadeOut(eq), FadeOut(axes), FadeOut(start_dot),
            FadeOut(s0_label), FadeOut(note), *[FadeOut(m) for m in path_mobjects],
        )

    # ACT 3 — terminal scatter collapses into the lognormal density.
   
    def act_3_terminal_distribution(self):
        title = Text("Where do all those paths end up?", font_size=34).to_edge(UP)
        self.play(Write(title))

        rng = np.random.default_rng(SEED)
        n_samples = 3000
        terminals = gbm_terminal_samples(S0, MU, SIGMA, T_MAX, n_samples, rng)

        axes = Axes(
            x_range=[20, 220, 20],
            y_range=[0, 0.022, 0.005],
            x_length=10.5,
            y_length=5,
            axis_config={"include_numbers": True},
        ).shift(DOWN * 0.4)
        x_label = axes.get_x_axis_label("S_T")
        self.play(Create(axes), Write(x_label))

        bins = np.linspace(20, 220, 41)
        counts, edges = np.histogram(terminals, bins=bins, density=True)
        bars = VGroup()
        for c, left, right in zip(counts, edges[:-1], edges[1:]):
            bar = Rectangle(
                width=axes.x_axis.unit_size * (right - left),
                height=max(axes.y_axis.unit_size * c, 0.001),
                stroke_width=0.5,
                stroke_color=WHITE,
                fill_color=BLUE,
                fill_opacity=0.7,
            )
            bar.move_to(axes.c2p((left + right) / 2, c / 2), aligned_edge=DOWN)
            bars.add(bar)

        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.02), run_time=2.5)

        def lognormal_pdf(s):
            if s <= 0:
                return 0
            mean_log = math.log(S0) + (MU - 0.5 * SIGMA ** 2) * T_MAX
            var_log = SIGMA ** 2 * T_MAX
            coeff = 1.0 / (s * math.sqrt(2 * math.pi * var_log))
            return coeff * math.exp(-((math.log(s) - mean_log) ** 2) / (2 * var_log))

        curve = axes.plot(lognormal_pdf, x_range=[21, 219, 1], color=YELLOW, stroke_width=4)
        curve_label = MathTex(
            r"f(S_T) = \frac{1}{S_T\sigma\sqrt{2\pi T}}"
            r"\exp\!\left(-\frac{(\ln S_T - \mu')^2}{2\sigma^2 T}\right)",
            font_size=28,
        ).to_edge(RIGHT)

        self.play(Create(curve), run_time=1.5)
        self.play(Write(curve_label))
        self.wait(2 )

        self.play(
            FadeOut(title), FadeOut(axes), FadeOut(x_label),
            FadeOut(bars), FadeOut(curve), FadeOut(curve_label),
        )


    # ACT 4 — Ito's Lemma derivation.
   
    def act_4_ito_lemma(self):
        title = Text("If S is random, what happens to f(S, t)?", font_size=32).to_edge(UP)
        self.play(Write(title))

        taylor = MathTex(
            r"df = \frac{\partial f}{\partial t}dt",
            r"+ \frac{\partial f}{\partial S}dS",
            r"+ \frac{1}{2}\frac{\partial^2 f}{\partial S^2}(dS)^2",
            r"+ \dots",
        ).scale(0.85).next_to(title, DOWN, buff=0.6)
        self.play(Write(taylor))
        self.wait(1)

        sub = MathTex(r"dS = \mu S\, dt + \sigma S\, dW").next_to(taylor, DOWN, buff=0.6)
        self.play(Write(sub))
        self.wait(1)

        rule = MathTex(r"(dt)^2 \to 0, \quad dt\, dW \to 0, \quad (dW)^2 \to dt").scale(0.75)
        rule.next_to(sub, DOWN, buff=0.6)
        self.play(Write(rule))
        self.wait(1.5)

        self.play(FadeOut(taylor), FadeOut(sub), FadeOut(rule))

        ito_final = MathTex(
            r"df = \left(\frac{\partial f}{\partial t}"
            r"+ \mu S\frac{\partial f}{\partial S}"
            r"+ \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}\right)dt"
            r"+ \sigma S\frac{\partial f}{\partial S}\, dW",
            font_size=36,
        ).next_to(title, DOWN, buff=1.0)
        box = SurroundingRectangle(ito_final, color=YELLOW, buff=0.3)
        label = Text("Ito's Lemma", font_size=30).next_to(box, DOWN, buff=0.4)

        self.play(Write(ito_final))
        self.play(Create(box), Write(label))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(ito_final), FadeOut(box), FadeOut(label))

  
    # ACT 5 — riskless hedge, cancellation, the PDE.
    
    def act_5_pde(self):
        title = Text("Hedge away the randomness", font_size=34).to_edge(UP)
        self.play(Write(title))

        portfolio = MathTex(r"\Pi = f - \frac{\partial f}{\partial S}\,S").next_to(title, DOWN, buff=0.6)
        portfolio_note = Text(
            "Hold the option, short df/dS shares - a self-financing hedge.", font_size=24
        ).next_to(portfolio, DOWN, buff=0.3)
        self.play(Write(portfolio), Write(portfolio_note))
        self.wait(1.5)
        self.play(FadeOut(portfolio_note))

        dpi_sub = MathTex(
            r"d\Pi = \left(\frac{\partial f}{\partial t}"
            r"+ \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}\right) dt"
        ).scale(0.85).next_to(portfolio, DOWN, buff=0.6)
        cancel_note = Text("The mu and dW terms cancel exactly.", font_size=24).next_to(dpi_sub, DOWN, buff=0.3)
        self.play(Write(dpi_sub), Write(cancel_note))
        self.wait(1.5)

        self.play(FadeOut(portfolio), FadeOut(dpi_sub), FadeOut(cancel_note))

        noarb = MathTex(r"\text{No randomness left} \;\Rightarrow\; d\Pi = r\Pi\, dt").next_to(title, DOWN, buff=0.7)
        noarb_note = Text(
            "A riskless portfolio must earn the riskless rate.", font_size=24
        ).next_to(noarb, DOWN, buff=0.4)
        self.play(Write(noarb), Write(noarb_note))
        self.wait(1.5)
        self.play(FadeOut(noarb_note))

        pde = MathTex(
            r"\frac{\partial f}{\partial t}"
            r"+ rS\frac{\partial f}{\partial S}"
            r"+ \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}"
            r"= rf",
            font_size=40,
        ).next_to(noarb, DOWN, buff=0.8)
        box = SurroundingRectangle(pde, color=YELLOW, buff=0.3)
        label = Text("The Black-Scholes PDE", font_size=30).next_to(box, DOWN, buff=0.4)
        self.play(Write(pde))
        self.play(Create(box), Write(label))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(noarb), FadeOut(pde), FadeOut(box), FadeOut(label))

    # ACT 6 — closed form, then Monte Carlo convergence.
    
    def act_6_price_and_convergence(self):
        title = Text("Solve the PDE with the call payoff boundary condition", font_size=28).to_edge(UP)
        self.play(Write(title))

        formula = MathTex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)").scale(1.1).next_to(title, DOWN, buff=0.6)
        d1d2 = MathTex(
            r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}},"
            r"\qquad d_2 = d_1 - \sigma\sqrt{T}"
        ).scale(0.75).next_to(formula, DOWN, buff=0.5)
        self.play(Write(formula), Write(d1d2))
        self.wait(1.5)

        true_price = bs_call_price(S0, K, R, SIGMA, T_MAX)
        price_box = SurroundingRectangle(formula, color=YELLOW, buff=0.3)
        price_value = MathTex(rf"C \approx {true_price:.3f}").next_to(d1d2, DOWN, buff=0.6)
        self.play(Create(price_box), Write(price_value))
        self.wait(1.5)

        self.play(FadeOut(formula), FadeOut(d1d2), FadeOut(price_box), FadeOut(price_value))

        subtitle = Text("Now check it the brute-force way", font_size=28).next_to(title, DOWN, buff=0.5)
        self.play(Write(subtitle))

        rng = np.random.default_rng(SEED)
        max_paths = 20000
        terminal = gbm_terminal_samples(S0, R, SIGMA, T_MAX, max_paths, rng)
        payoffs = np.exp(-R * T_MAX) * np.maximum(terminal - K, 0.0)
        running_mean = np.cumsum(payoffs) / np.arange(1, max_paths + 1)

        sample_points = np.unique(np.logspace(1, math.log10(max_paths), 200).astype(int))
        sample_points = sample_points[sample_points < max_paths]
        ns = sample_points
        means = running_mean[sample_points - 1]

        axes = Axes(
            x_range=[0, math.log10(max_paths), 1],
            y_range=[true_price - 4, true_price + 4, 1],
            x_length=10,
            y_length=5,
            axis_config={"include_numbers": False},
        ).shift(DOWN * 0.4)
        x_label = axes.get_x_axis_label(r"\log_{10}(\text{paths})")
        y_label = axes.get_y_axis_label("\\hat{C}")
        self.play(FadeOut(subtitle), Create(axes), Write(x_label), Write(y_label))

        target_line = DashedLine(
            axes.c2p(0, true_price), axes.c2p(math.log10(max_paths), true_price), color=YELLOW
        )
        target_label = MathTex(rf"C_{{BS}} = {true_price:.3f}", font_size=28, color=YELLOW)
        target_label.next_to(target_line, UP, buff=0.15).align_to(target_line, RIGHT)
        self.play(Create(target_line), Write(target_label))

        graph_points = [axes.c2p(math.log10(n), m) for n, m in zip(ns, means)]
        mc_curve = VMobject(color=BLUE, stroke_width=3)
        mc_curve.set_points_as_corners([graph_points[0]])

        tracker = ValueTracker(0)

        def update_curve(mob):
            i = int(tracker.get_value())
            i = max(1, min(i, len(graph_points)))
            mob.set_points_as_corners(graph_points[:i])

        mc_curve.add_updater(update_curve)
        moving_dot = always_redraw(
            lambda: Dot(
                graph_points[max(0, min(int(tracker.get_value()), len(graph_points) - 1))],
                color=RED, radius=0.07,
            )
        )
        running_label = always_redraw(
            lambda: MathTex(
                rf"\hat{{C}}_n \approx {means[max(0, min(int(tracker.get_value()), len(means) - 1))]:.3f}",
                font_size=30,
            ).to_corner(UR).shift(DOWN * 1.2)
        )

        self.add(mc_curve, moving_dot, running_label)
        self.play(tracker.animate.set_value(len(graph_points)), run_time=7, rate_func=linear)
        self.wait(1)

        closing = Text("Two completely different routes. Same number. by the cosmic cube ", font_size=28).to_edge(DOWN)
        self.play(Write(closing))
        self.wait(2)
