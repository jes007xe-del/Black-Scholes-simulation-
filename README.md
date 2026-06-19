# Black-Scholes-simulation-
# Visualizing the Black-Scholes Model with Manim

A beautifully animated, step-by-step visual exploration of the mathematical and algorithmic foundations behind Wall Street's most famous equation: the **Black-Scholes Model**. 

This project contrasts the traditional **Analytical Route** (solving the Partial Differential Equation via Ito's Lemma) against an **Empirical Route** (brute-force Monte Carlo simulations), demonstrating how both converge on the exact same theoretical option price.

---

##  Project Overview

The animation takes the viewer on a conceptual journey through quantitative finance:
1. **The Foundation of Randomness:** Simulating a standard Wiener Process ($W_t$) to represent market noise.
2. **Asset Price Modeling:** Transitioning pure noise into **Geometric Brownian Motion** ($dS_t = \mu S_t dt + \sigma S_t dW_t$).
3. **Probability Distributions:** Watching chaotic random paths flatten out into a predictable Log-Normal Distribution.
4. **The No-Arbitrage Principle:** Utilizing Ito's Lemma to construct a risk-neutral, perfectly hedged portfolio, revealing the Black-Scholes Partial Differential Equation.
5. **Calculus vs. Brute Force:** Comparing the exact mathematical formula directly against a thousands-path Monte Carlo simulation as they converge beautifully to the same value (`12.336`).

---

##  Prerequisites & Installation

### System Dependencies
Because this project relies on **Manim**, you must have the following system-level dependencies installed on your machine:
* **Python 3.9 - 3.12**
* **FFmpeg** (for video rendering)
* **LaTeX** (e.g., TeX Live or MiKTeX, for rendering mathematical equations)



