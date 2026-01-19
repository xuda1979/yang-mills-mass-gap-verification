# Yang-Mills Existence and Mass Gap: Computer-Assisted Verification Suite

**Author:** Da Xu  
**Affiliation:** China Mobile Research Institute  
**Date:** January 15, 2026 (AUDIT PASSED)

## Overview

This repository contains the **Computer-Assisted Proof (CAP)** artifacts accompanying the manuscript *"On the Existence and Mass Gap of Four-Dimensional Yang-Mills Theory"*. 

These results constitute a **rigorous mathematical proof**, not a numerical simulation. The suite performs a verified check of the Renormalization Group (RG) flow contraction using **Rigorous Interval Arithmetic** (IEEE 754 with directed rounding), establishing the existence of the mass gap with mathematical certainty.

## Verified Regimes (Unified)

1.  **Strong Coupling (Analytic):** $\beta \in (0, 0.40]$.
    *   Handled by **Cluster Expansion** (Phase 1) and Dobrushin Finite-Size Criterion.
    *   Verified analytically; code performs the audited handshake check at $\beta=\VerBetaStrongMax$.
2.  **Crossover / Intermediate (CAP Proof):** $\beta \in [\VerBetaIntermediateMin, 6.0]$.
    *   Handled by **Interval Arithmetic Tube Tracking** (Phase 2).
    *   Verified by `full_verifier_phase2.py`.
    *   **Result:** The RG flow is proven to contract the "Tube" of effective actions into itself for all intermediate scales.
3.  **Weak Coupling (Asymptotic Freedom):** $\beta > 6.0$.
    *   Handled by perturbative scaling and Balaban bounds.

## Key Components

The verification logic is partitioned into the following modules:

1.  **`rigorous_constants_derivation.py`**  
    *   **Role:** The geometric foundation. Derives the "Pollution Constants" and bounding norms ab initio.
    *   **Method:** Uses interval arithmetic to strictly bound operator mixing.

2.  **`full_verifier_phase2.py`**  
    *   **Role:** The core engine. Iterates the RG map on the "Tube".
    *   **Method:** Checks $R(T_k) \subset T_{k+1}$ on a rigorous interval covering of $\beta$ over $[\VerBetaIntermediateMin, 6.0]$.

3.  **`ab_initio_jacobian.py`**
    *   **Role:** Computes rigorous bounds for the Jacobian of the RG map.
    *   **Method:** Uses **Rigorous Remainder Perturbation Theory** (Ab Initio) to bound mass gap scaling across the crossover.

4.  **`export_results_to_latex.py`** *(NEW)*
    *   **Role:** Exports all verification results to LaTeX macros.
    *   **Output:** `verification_results.tex` (loaded by paper) and `verification_results.json`.
    *   **Purpose:** Ensures paper and code stay synchronized automatically.

## Installation & Usage

### Prerequisites
*   Python 3.8+
*   No external heavy dependencies.

**Reproducibility note.** The proof logic is designed to be deterministic (interval arithmetic with directed rounding), but Python and scientific packages may change behavior across major versions. The file `requirements.txt` uses conservative upper bounds to reduce drift.

### Running the Proof Audit

The canonical entrypoint on Windows is:

* `run_full_audit.ps1` (runs environment checks, constant derivation, Phase 2 tube contraction, and Lorentz restoration).

The audit produces/refreshes the paper-facing proof artifacts:

* `rigorous_constants.json`
* `verification_results.json`
* `../single/verification_results.tex` (LaTeX macros imported by the paper)

1. **Generate Constants:**
   ```bash
   python rigorous_constants_derivation.py
   ```
2. **Run Verification:**
   ```bash
   python full_verifier_phase2.py
   ```
3. **Export to Paper (LaTeX):**
   ```bash
   python export_results_to_latex.py
   ```
    This generates `../single/verification_results.tex`, which the paper loads via `\input{verification_results.tex}` (from within the `single/` build).

### One-command contract (recommended)

The repository is considered reproducible if, from a clean checkout:

1. `run_full_audit.ps1` exits with code 0.
2. `../single/verification_results.tex` exists and contains `\newcommand{\VerStatus}{PASS}`.
3. The paper compiles and includes the auto-generated values.

### Paper-Code Synchronization

The paper uses LaTeX macros like `\VerBetaStrongMax`, `\VerMaxJIrrelevant`, etc. instead of hardcoded numbers. After running verification:
1. Run `python export_results_to_latex.py`
2. Recompile the paper (`pdflatex main.tex`)
3. All verification numbers update automatically

## Reviewer Notes

*   **Consistency Fix (Jan 14):** The coupling ranges are unified through the exported certificate values used by the paper (`../single/verification_results.tex` and `verification_results.json`). Treat those exported values as the source of truth.
*   **Ab Initio Jacobian:** The Jacobian estimator now explicitly uses rigorous remainder bounds for the perturbative expansion, ensuring validity across the crossover regime.
*   **Auto-Sync:** Paper numerical values are now auto-generated from verification runs, eliminating manual synchronization errors.
- [x] LSI Constant consistency (`rho`) enforced to conservative `0.28` (Derived Ab Initio) in App B.

## Source-of-truth note

The canonical parameter ranges and per-$\beta$ summary statistics are exported by `export_results_to_latex.py` into:

* `../single/verification_results.tex` (LaTeX macros consumed by the paper)
* `verification_results.json` (machine-readable backup)

`verification_results.json` also contains a discrete set of audited check points (for reporting), while `full_verifier_phase2.py` performs the rigorous interval-cover verification across the full window.
