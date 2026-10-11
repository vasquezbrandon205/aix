# AXIOM-X v2 — Modernization Plan

## What Changed in the Field (Dec 2025 → Oct 2026)

| Approach | 2025 Status | 2026 Status |
|---|---|---|
| Pure evolutionary SR | Standard (PySR, v1) | Legacy baseline |
| LLM-guided SR | Emerging (LLM-SR 2024) | **SOTA** (InsightSR 95% Feynman, LLM-ACES SOTA ODEBench) |
| LLM-evolved operators | Novel | Published (LLM-Meta-SR) |
| Closed-loop discovery | Rare | SOTA (LLM-ACES active acquisition) |

## v2 Architecture: "LLM Proposes, GPU Disposes"

```
┌─────────────────────────────────────────────────┐
│  LLM GUIDANCE LAYER (slow, smart, infrequent)   │
│  - Semantic seeding from data analysis          │
│  - Structural mutation proposals                │
│  - Physics plausibility critique                │
│  - Operator-level prior induction               │
└──────────────────┬──────────────────────────────┘
                   │ every K generations
┌──────────────────▼──────────────────────────────┐
│  GPU EVOLUTION CORE (fast, dumb, massive)        │
│  - 131k population batched eval (from v1)        │
│  - Tournament selection (from v1)               │
│  - Multi-objective fitness: MSE + λ·complexity  │
│  - Deterministic checkpointing (from v1)        │
└─────────────────────────────────────────────────┘
```

### Why This Works

The LLM is expensive (~seconds per call). The GPU evaluates 131k candidates
in milliseconds. The hybrid uses each for what it's good at:
- LLM runs every 50 generations: proposes 10-100 structural ideas
- GPU runs every generation: evaluates 131k candidates including LLM proposals
- Net effect: search is *guided* by physics intuition, *powered* by brute force

This mirrors InsightSR's architecture but with a 10-100x larger evaluation
budget per generation (their PySR backend vs our 131k batched GPU eval).

## New Components

### 1. SemanticSeeding
Analyze trajectory statistics BEFORE evolution starts:
- Growth/decay → seed exponential terms
- Oscillation → seed sinusoidal terms
- Saturation → seed tanh/sigmoid terms
- Polynomial trend → seed polynomial terms

v1 started from pure Gaussian noise. v2 starts from data-informed priors.

### 2. LLMGuide (pluggable)
Abstract interface with two implementations:
- `HeuristicGuide`: Works offline. Uses trajectory stats + physics heuristics
  to propose structural mutations. No API key needed.
- `APIGuide`: Stub for OpenAI/Anthropic/local LLM. Drop in any
  OpenAI-compatible endpoint.

### 3. MultiObjectiveFitness
v1 optimized MSE only → bloat risk. v2 uses:
```
fitness = MSE + λ · complexity_penalty
```
where complexity = number of active terms. Produces Pareto frontier of
accuracy vs. simplicity (the actual goal of symbolic regression).

### 4. CriticLoop
Every K generations, the guide reviews top-10 elites and:
- Flags physically implausible terms (e.g., dimensionally inconsistent)
- Proposes structural alternatives
- Injects 1-5% new blood into population from proposals

### 5. ActiveAcquisition (future)
LLM-ACES showed SOTA gains from disagreement-driven data collection.
v2 architecture supports this: when top hypotheses disagree on predictions
for unsampled regions, flag for new data. Stub included.

## What We Keep From v1
- GPU batched evaluation (the moat)
- 131k population scale
- Tournament selection + elitism
- Full deterministic checkpointing (CPU+CUDA RNG)
- Signal-safe autosave

## What We Drop
- Pure random Gaussian initialization → semantic seeding
- MSE-only fitness → multi-objective
- Fixed mutation schedule → adaptive + LLM-guided structural mutation

## Benchmark Targets
- Feynman 100-problem benchmark (InsightSR hit 95% exact recovery)
- ODEBench (LLM-ACES SOTA)
- v1's own synthetic trajectory (regression check: v2 must match or beat)
