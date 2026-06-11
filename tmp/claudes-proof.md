Yes. The intuition we've built turns into a genuine theorem once you commit to a few assumptions that make the "thin bottleneck" precise. Here is a paper-ready version. The whole argument hinges on one empirically checkable assumption (cross-day contraction, $L<1$), which I flag explicitly.

## Setup

Let $s_n \in \mathbb{R}^d$ be the inter-day global state (the only quantity crossing day boundaries; $d$ small). One day's evolution of this state, under both policies and all day-$n$ randomness $\omega_n$, is a map

$$s_n = F(s_{n-1}, \omega_n), \qquad n = 1,\dots,N, \quad s_0 \text{ fixed.}$$

$\omega_n = (\xi_n, \eta_n)$ collects **exogenous** (environment-seed) noise $\xi_n$ and **endogenous** (policy action-sampling) noise $\eta_n$. The sequential rollout iterates $F$ in order. The Picard scheme runs $K$ Jacobi sweeps with **common random numbers** (seeds fixed across sweeps), from an initial guess $\hat s_n^{(0)}$:

$$\hat s_n^{(k+1)} = F\!\left(\hat s_{n-1}^{(k)}, \omega_n\right), \qquad \hat s_0^{(k)} = s_0.$$

## Assumptions

- **(A1) Independent innovations.** The per-day randomness $\omega_1,\dots,\omega_N$ is mutually independent. (This is exactly your seeding construction — independent per-day seeds *and* independent per-day action sampling.)
- **(A2) Cross-day contraction.** $F$ is $L$-Lipschitz in its state argument, uniformly in $\omega$, with $L<1$: $\|F(s,\omega)-F(s',\omega)\|\le L\|s-s'\|$. Intuitively, each day partially "forgets" its entering global state because the environment resets. *This is the crux assumption.*
- **(A3) Non-inflating guess.** The initial guess is no more dispersed than the quantity it estimates: $\operatorname{Cov}(\hat s_m^{(0)}) \preceq \operatorname{Cov}(s_m^{\mathrm{seq}})$ in the PSD order. Any shrinkage / conditional-mean predictor satisfies this.
- **(A4) Lipschitz return.** The high-level return is $L_g$-Lipschitz in the boundary path $(s_n,\dots,s_N)$.

For the exact covariance statements I linearize $F$ about the nominal path, $s_n = A s_{n-1} + b + \epsilon_n$ with $\|A\|=L$, $\operatorname{Cov}(\epsilon_n)=\Sigma=\Sigma_{\text{exo}}+\Sigma_{\text{endo}}$; the bias result and an Efron–Stein variance *bound* hold for the nonlinear $F$ directly (remarked below).

## Lemma 1 (Bias contracts geometrically; innovations cancel)

Unrolling $K$ sweeps gives, for $n\ge K$,

$$\hat s_n^{(K)} = A^{K}\hat s_{n-K}^{(0)} + \sum_{j=0}^{K-1} A^{j}\big(b+\epsilon_{n-j}\big),$$

and the sequential path satisfies the same recursion with $\hat s_{n-K}^{(0)}$ replaced by $s_{n-K}^{\mathrm{seq}}$ and *the identical* $\epsilon$'s (CRN). Subtracting, the innovations cancel exactly:

$$\boxed{\;\hat s_n^{(K)} - s_n^{\mathrm{seq}} = A^{K}\big(\hat s_{n-K}^{(0)} - s_{n-K}^{\mathrm{seq}}\big)\;}\quad\Rightarrow\quad \big\|\hat s_n^{(K)} - s_n^{\mathrm{seq}}\big\| \le L^{K} e_0,$$

with $e_0 := \max_m \|\hat s_m^{(0)} - s_m^{\mathrm{seq}}\|$. So the per-realization error is purely the propagated guess error, contracted by $L^K$ — which is why $K=1\!-\!2$ already tracks the sequential rollout under (A2).

## Proposition (Variance reduction)

A $K$-sweep Picard estimate depends only on the **most recent $K$ innovations**; the deep history $\epsilon_1,\dots,\epsilon_{n-K}$ is replaced by the low-variance guess. The covariances are

$$\operatorname{Cov}(s_n^{\mathrm{seq}}) = \sum_{j=0}^{n-1} A^{j}\Sigma (A^\top)^{j}, \qquad \operatorname{Cov}(\hat s_n^{(K)}) = A^{K}\operatorname{Cov}(\hat s_{n-K}^{(0)})(A^\top)^{K} + \sum_{j=0}^{K-1} A^{j}\Sigma (A^\top)^{j},$$

and their difference collapses to a single conjugated term:

$$\boxed{\;\operatorname{Cov}(s_n^{\mathrm{seq}}) - \operatorname{Cov}(\hat s_n^{(K)}) = A^{K}\Big[\operatorname{Cov}(s_{n-K}^{\mathrm{seq}}) - \operatorname{Cov}(\hat s_{n-K}^{(0)})\Big](A^\top)^{K} \;\succeq\; 0\;}$$

by (A3). **The variance gap equals the propagated dispersion gap between truth and guess** — strictly positive whenever $A$ is non-degenerate, $K<n$, and the guess is strictly tighter. (Nonlinear version: Efron–Stein with (A2) gives $\operatorname{Var}(s_n^{\mathrm{seq}}) \le c\sum_{i=0}^{n-1}L^{2i}$ vs. $\operatorname{Var}(\hat s_n^{(K)}) \le c\sum_{i=0}^{K-1}L^{2i} + L^{2K}\operatorname{Var}(\hat s^{(0)})$ — same truncation structure as a rigorous upper bound.)

## Corollary 1 (Critic MSE improves)

By (A4) and the delta method, target variance is monotone in boundary covariance, so $\operatorname{Var}(Y^{\text{Picard}}) \le \operatorname{Var}(Y^{\mathrm{seq}})$ with the reduction inherited from the PSD gap above; meanwhile the target **bias** is $\le L_g\, L^{K} e_0/(1-\gamma)$ by Lemma 1. The critic's mean-squared error decomposes as variance ($\downarrow$ by an $O(1)$ gap) plus bias-squared ($O(L^{2K}e_0^2)$, second-order small). Hence under (A2)–(A3) the net MSE strictly improves, the irreducible loss floor drops, and gradient noise falls — reproducing the lower, smoother critic loss and faster convergence observed.

## Corollary 2 (Asymmetry: low level unaffected)

The high-level return integrates the entire downstream boundary path $s_n,\dots,s_N$, so it carries the *compounded* boundary covariance $\sum_j A^j\Sigma(A^\top)^j$ that the Proposition shrinks. The low-level policy resets each day and conditions on $s_{n-1}$ as an *input feature*; its return spans only the within-day horizon and does not compound across days. The boundary variance is therefore the high level's integrand but merely the low level's conditioning variable, so the reduction benefits the high-level critic specifically — the asymmetry you report.

## Remark on regimes

The whole result is governed by (A2). If instead $L>1$ (cross-day amplification / chaos), sequential returns have variance blowing up like $\sum_j L^{2j}$, and Picard's truncation to the last $K$ innovations *caps* this — the variance benefit grows, though bias no longer vanishes and convergence slows; the clean $K=1\!-\!2$ regime is specifically the $L<1$, thin-bottleneck case. I'd recommend estimating $L$ empirically (e.g., the spectral radius of the day-to-day global-state Jacobian, or a finite-difference sensitivity of $s_n$ to a perturbation of $s_{n-1}$) and reporting it — a measured $L<1$ is the direct evidence the proof's hypothesis holds in your system.

---

Want me to typeset this into an IEEEtran LaTeX file (or Typst) with the lemma/proposition environments and proofs spelled out in full? I can also fold in the empirical $L$-estimate as a short validation subsection.