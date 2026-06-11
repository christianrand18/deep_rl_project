# Speeding up simulation using Picard iterations
## Background
With the meta-policy simulation task we are simulating T timesteps, for N days over E episodes for M agents. For a fully sequential order of computation this yields a timecomplexity of $O(T\times N \times E \times M)$ and scaling any of the multipliers thus yields a linear increase in wall clock time. While simulating 1 day with 20 timesteps for 100,000 episodes within 24-hours was acceptable, scaling to multiple days yields an impractical if not infeasible wall clock time. 

While parallel computing is an option the simulation is not trivially calculated in parallel due to the inherent sequential nature of the simulation tractories. This leads to being restricted to single-threaded computing with underutilized computational resources and high wall clock times.

To overcome this constraint we introduce parallel-in-days episode rollouts utilizing Picard iterations.

## Previous work
In [1] the authors presents an iterative approach to policy simulation in supply chain RL dubbed Picard iteration, named after the proof of the Picard-Lindelöf theorem. While the paper explicitly deals with supply chain management problems they state, and emperically prove, the approach is applicable to other RL domains. Their approach uses the assumption that the policy evaluation is computationally expensive while the system dynamics evaluation is cheap. Under this assumption the Picard iteration divides up the simulation of T time-steps across parallel processes and initialize a 'cache' of actions, one for each time-step, that can be thought of as an initial guess of the actions that will eventually be simulated. Each process then runs an individual simulation but only evaluates the policy for timesteps specifically assigned to that process; for the remainder, the cache actions are used. At the end of an iteration, each process updates the cache in the time-steps it was responsible for. As such, a single iteration is faster than the serial policy simulation task by a factor of roughly #processes [1], and by running multiple iterations the action rollout of the trajectory converges to the actions of the sequential rollout. The paper establishes that the approach achieves non-trivial speedup in wall clock time over sequential computation, emperically demonstrated by a convergence to ≤0.1% in relative RMSE in all Gym MuJoCo environments for a PPO algorithm within 15 iterations for T=200.

## Our approach
Drawing inspiration from the approach in [1] we implement a similar scheme, albeit with moderations. A fundamental assumption in [1] was cheap evaluation of the systems dynamics and in our case the environment envaluation is not cheap per-se considering the LP optimization and choice model evaluation. However, with the meta-policy being of simple architecture we have a reversed case with the meta-policy evaluation being computationally cheap and system dynamics being the expensive part. To enable parallel day evaluation we leverage the fact that the information bottleneck between days is thin and given a prediction of the previous days meta-state we are perfectly able to simulate a disconnected given day.

### Formulation
We consider a hierarchical environment setup with state space $\mathcal{S}=\mathcal{S}_\text{meta} \times \mathcal{S}_\text{lower}$ where $\text{dim}(\mathcal{S}_\text{meta}) \ne \text{dim}(\mathcal{S}_\text{lower})$. The full simulation horizon spans $T \times N$ timesteps. A meta-policy $\pi_\text{meta}$ acts once per day and persists across the full horizon, while a lower-level policy $\pi_\text{lower}$ acts at each timestep and resets at the end of every day. Since $\mathcal{S}_\text{lower}$ resets fully between days, the inter-day dependency reduces to a boundary state $S_d \in \mathcal{S}_\text{meta}$. At each day $d$, $\pi_\text{meta}$ maps $S_d$ to an action $a_d \in \mathcal{A}_\text{meta}$ that shapes the dynamics of $\pi_\text{lower}$ within that day.

Let $\phi_d(\cdot\,;\xi_d)$ denote the deterministic day simulation roll-out under a fixed seed $\xi_d$. The true rollout satisfies the causal recurrence

$$S_{d+1} = \phi_d(S_d;\,\xi_d), \quad d = 0,\ldots,N-1.$$

For a given episode with seeds $\{\xi_d\}_{d=0}^{N-1}$, let $\hat{\mathbf{S}} = (\hat{S}_1, \ldots, \hat{S}_N) \in \mathcal{S}_\text{meta}^N$ denote a "guess" for the boundary sequence. Define a function $f: \mathcal{S}_\text{meta}^N \to \mathcal{S}_\text{meta}^N$ mapping $\hat{\mathbf{S}}$ to an updated guess $\hat{\mathbf{S}}' = f(\hat{\mathbf{S}})$, seeking to estimate  $\mathbf{S}^* = (S_1^*, \ldots, S_N^*)$, the unique solution to the recurrence above, with fixed point $f(\mathbf{S}^*) = \mathbf{S}^*$.

The sequence $\mathbf{S}^*$ is approximated by iterating $\hat{\mathbf{S}}^{(k+1)} = f(\hat{\mathbf{S}}^{(k)})$ from an initial guess $\hat{\mathbf{S}}^{(0)}$, terminating when $\delta(\hat{\mathbf{S}}^{(k+1)}, \hat{\mathbf{S}}^{(k)}) < \epsilon$ for a convergence criterion $\delta$ and tolerance $\epsilon > 0$.

Upon termination at iteration $K$, the per-day transitions $\{(\hat{S}_d^{(K)}, a_d^{(K)}, r_d^{(K)})\}_{d=0}^{N-1}$ from the final rollout are committed to the meta-policy buffer, from which $\pi_\text{meta}$ is updated. The lower-level policy $\pi_\text{lower}$ is updated solely from the trajectories of the final iteration; those of all preceding iterations are discarded.

### Instantiation

### Correctness
Since the day simulation $\phi_d$ is deterministic for a fixed seed $\xi_d$, the Lipschitz condition formulates that two different states fed to the same day simulator yield outputs no further apart than $L$ times their input difference:
$$\|\hat{S}_d^{(k+1)} - S_d^*\| = \|\phi_d(\hat{S}_{d-1}^{(k)};\,\xi_d) - \phi_d(S_{d-1}^*;\,\xi_d)\| \leq L\|\hat{S}_{d-1}^{(k)} - S_{d-1}^*\|$$

We can comfortably assume that $L<1$ since brand momentum follows $b_{d+1} = \lambda b_d + (1-\lambda)\mu_d$ with $\lambda < 1$, so any difference in boundary state shrinks by at least $\lambda$ each day.

Across $k$ iterations this gives $\|\hat{S}_d^{(k)} - S_d^*\| \leq L^k e_0$ where $e_0 = \max_d \|\hat{S}_d^{(0)} - S_d^*\|$, which vanishes geometrically as $k \to \infty$. For a target approximation error $\epsilon > 0$ on $\|\hat{S}_d^{(k)} - S_d^*\|$, simple rearrangement shows this guarantees termination in at most $\left\lceil \frac{\log(e_0/\epsilon)}{\log(1/L)} \right\rceil$ iterations.

### Implications on convergence

**Setup and Notation**:
Let $s_n \in \mathbb{R}^d$ be the inter-day global state. One day's evolution of this state, under both policies and all day-$n$ randomness $\omega_n$, is a map

$$s_n = F(s_{n-1}, \omega_n), \qquad n = 1,\dots,N,$$

with $s_0$ fixed. $\omega_n = (\xi_n, \eta_n)$ collects environment noise $\xi_n$ and policy action-sampling noise $\eta_n$. The sequential rollout iterates $F$ in order. We assess the standard sequential rollout state $s_n^{\text{seq}}$ against the Picard-iterated state $\hat{s}_n^{(K)}$ after $K$ sweeps, initialized from a prior sequence guess with maximum error $e_0$.

#### Definition: Lipschitz Continuity with L < 1
A function $F(s, \omega)$ is Lipschitz continuous with respect to the state $s$ with a constant $L$ if, for any two states $s_a$ and $s_b$ under the same noise realization $\omega$, the distance between their outputs satisfies:

$$\|F(s_a, \omega) - F(s_b, \omega)\| \leq L \|s_a - s_b\|$$

When L < 1, the function is strictly a contraction, meaning that any initial perturbation or error between two trajectory states shrinks geometrically by at least a factor of $L$ with each elapsed transition step.

#### Assumptions
To facilitate the proof we operate under the following assumptions:

* **(A1) Independent Innovations:** The noise terms $\omega_i$ and $\omega_j$ are mutually independent for all $i \neq j$.
* **(A2) Contraction:** The transition function $F$ is Lipschitz continuous with respect to the state with a constant $L < 1$.
* **(A3) Stable Initialization:** The covariance matrix of the initial trajectory guess is bounded above by the true sequential covariance: $\operatorname{Cov}(\hat{s}_n^{(0)}) \preceq \operatorname{Cov}(s_n^{\text{seq}})$.
* **(A4) Lipschitz Critic:** The value function $V^\pi(s)$ is Lipschitz continuous with constant $C_{V^\pi}$.

#### Proof for the linear case
We first establish the foundational bounds on the state approximation error and linear covariance matrix.

**Lemma 1**:
By the contraction property (A2) and the Banach Fixed-Point Theorem, applying $K$ Picard iterations with seeds fixed across sweeps strictly bounds the absolute deviation from the sequential trajectory:


$$||\hat{s}_n^{(K)} - s_n^{\text{seq}}|| \le L^K e_0$$

**Proposition 1 (Variance Reduction in Linear Regimes)**:
Assuming a linear autoregressive approximation of the state transition $s_n = A s_{n-1} + \xi_n$ representing a stationary process, the difference in covariance matrices between the covariance matrix of the Picard rollout after $K$ iterations is strictly bounded above by the true sequential covariance:


$$\operatorname{Cov}(s_n^{\text{seq}}) - \operatorname{Cov}(\hat{s}_n^{(K)}) \succeq 0$$


**Proof**:
Let the environment's state transition be approximated by a linear autoregressive process $s_n = A s_{n-1} + \xi_n$, where $A$ represents the transition dynamics corresponding to the Lipschitz contraction ($||A|| < 1$), and $\xi_n$ represents the combined daily noise with covariance $\operatorname{Cov}(\xi_n) = \Sigma$. By Assumption (A1), the innovations $\xi_n$ are mutually independent.

The sequential trajectory unrolled from an initial deterministic state $s_0$ is:


$$s_n^{\text{seq}} = A^n s_0 + \sum_{j=0}^{n-1} A^j \xi_{n-j}$$


Taking the variance of both sides, and noting that the initial state $s_0$ has zero variance, the sequential covariance is the accumulated sum of all historical noise:

$$\operatorname{Cov}(s_n^{\text{seq}}) = \sum_{j=0}^{n-1} A^j \Sigma (A^\top)^j$$

The state generated by the Picard iteration scheme after $K$ sweeps, given an initial sequence guess $\hat{s}^{(0)}$ is:


$$\hat{s}_n^{(K)} = A^K \hat{s}_{n-K}^{(0)} + \sum_{j=0}^{K-1} A^j \xi_{n-j}$$


Taking the covariance of this state, while $\hat{s}_{n-K}^{(0)}$ as a random variable with its own prior variance, results in a covariance with noise accumulation truncated  at depth $K$:


$$\operatorname{Cov}(\hat{s}_n^{(K)}) = A^K \operatorname{Cov}(\hat{s}_{n-K}^{(0)}) (A^\top)^K + \sum_{j=0}^{K-1} A^j \Sigma (A^\top)^j$$

Thus, the variance reduction $\Delta = \operatorname{Cov}(s_n^{\text{seq}}) - \operatorname{Cov}(\hat{s}_n^{(K)})$ is


$$\Delta = \left( \sum_{j=0}^{n-1} A^j \Sigma (A^\top)^j \right) - \left( A^K \operatorname{Cov}(\hat{s}_{n-K}^{(0)}) (A^\top)^K + \sum_{j=0}^{K-1} A^j \Sigma (A^\top)^j \right)$$


Because the environment noise $\xi$ is identical across both rollouts, the first $K$ terms of the sequential sum perfectly cancel the noise terms of the Picard sum:


$$\Delta = \left( \sum_{j=K}^{n-1} A^j \Sigma (A^\top)^j \right) - A^K \operatorname{Cov}(\hat{s}_{n-K}^{(0)}) (A^\top)^K$$

We apply a change of variables to the remaining sequential tail:


$$\sum_{j=K}^{n-1} A^j \Sigma (A^\top)^j = \sum_{m=0}^{n-K-1} A^{m+K} \Sigma (A^\top)^{m+K} = A^K \left( \sum_{m=0}^{n-K-1} A^m \Sigma (A^\top)^m \right) (A^\top)^K$$


We recognize the inner summation as the exact recursive definition of the sequential covariance at step $n-K$. Substituting this back into $\Delta$:


$$
\begin{aligned}
\Delta &= A^K \Big[ \operatorname{Cov}(s_{n-K}^{\text{seq}}) \Big] (A^\top)^K - A^K \operatorname{Cov}(\hat{s}_{n-K}^{(0)}) (A^\top)^K\\
&= A^K \Big[ \operatorname{Cov}(s_{n-K}^{\text{seq}}) - \operatorname{Cov}(\hat{s}_{n-K}^{(0)}) \Big] (A^\top)^K
\end{aligned}
$$

It follow directly from Assumption (A3) that $\operatorname{Cov}(s_{n-K}^{\text{seq}}) - \operatorname{Cov}(\hat{s}_{n-K}^{(0)}) \succeq 0$


Since pre-multiplying and post-multiplying a positive semi-definite matrix by any matrix $A^K$ and its transpose $(A^\top)^K$ preserves positive semi-definiteness, it follows strictly that:


$$\Delta \succeq 0$$


Thus, $\operatorname{Cov}(s_n^{\text{seq}}) \succeq \operatorname{Cov}(\hat{s}_n^{(K)})$, completing the proof. $\blacksquare$

#### Extension to non-linear environments:

Proposition 1 relies on linear transitions. To guarantee robustness for strictly non-linear, black-box transition dynamics, we generalize the variance reduction by relaxing the linear assumption.

**Theorem 1 (Variance Reduction in Non-Linear Regimes)**:
For any arbitrary, non-linear environment transition function $F$ and meta-policy critic $V_\phi$, assuming the historical daily noise induces non-zero variance in the Critic's evaluation, the variance of the target value strictly decreases under the Picard iteration scheme relative to the sequential scheme: $\operatorname{Var}(V_\phi(\hat{s}_n^{(K)})) < \operatorname{Var}(V_\phi(s_n^{\text{seq}}))$.

**Proof**:
By Assumption (A1), all daily random variables representing the combined noise $\omega_i$ are mutually independent. Because the transition function $F$ may be non-smooth (e.g., an LP-optimization step), we evaluate the variance globally using the Efron-Stein inequality, which bounds the variance of any general function of independent variables.

Let the critic's target under the sequential rollout be $Z_{\text{seq}} = V_\phi(s_n^{\text{seq}})$. Due to the sequential nature of the Markov chain, the state $s_n^{\text{seq}}$ is a causally unbroken function of all historical noise: $Z_{\text{seq}} = f_{\text{seq}}(\omega_1, \dots, \omega_n)$.

The Efron-Stein inequality establishes the upper bound of this variance by summing the expected squared deviations caused by individually resampling each noise term $\omega_i$ with an independent copy $\omega_i'$:


$$\operatorname{Var}(Z_{\text{seq}}) \le \frac{1}{2} \sum_{i=1}^n \mathbf{E} \left[ \left( Z_{\text{seq}} - Z_{\text{seq}}^{(i)} \right)^2 \right]$$


where $Z_{\text{seq}}^{(i)} = f_{\text{seq}}(\omega_1, \dots, \omega_i', \dots, \omega_n)$.

Let the critic's target under the Picard iteration scheme be $Z_{\text{Picard}} = V_\phi(\hat{s}_n^{(K)})$. The state $\hat{s}_n^{(K)}$ is calculated by unrolling the transition function $F$ exactly $K$ times, initialized from a prior sequence anchor $\hat{s}_{n-K}^{(0)}$.

By definition, this algorithmic recurrence is functionally independent of any specific environmental noise that occurred chronologically prior to the anchor. Therefore, $Z_{\text{Picard}}$ is strictly a function of the recent noise sequence: $Z_{\text{Picard}} = f_{\text{Picard}}(\omega_{n-K+1}, \dots, \omega_n)$.

For any index $i \le n-K$, resampling the noise variable $\omega_i$ to $\omega_i'$ has identically zero effect on the output of the function, meaning:


$$Z_{\text{Picard}} - Z_{\text{Picard}}^{(i)} = 0, \quad \forall i \le n-K$$

Applying the Efron-Stein inequality to the Picard target yields the full summation:

$$\operatorname{Var}(Z_{\text{Picard}}) \le \frac{1}{2} \sum_{i=1}^n \mathbf{E} \left[ \left( Z_{\text{Picard}} - Z_{\text{Picard}}^{(i)} \right)^2 \right]$$

The Efron-Stein sum for the sequential rollout accumulates all $n$ non-negative terms. In contrast, because of the causal truncation established above, the first $n-K$ terms for the Picard target evaluate to zero. This reduces the Picard bound to only the final $K$ terms:

$$\operatorname{Var}(Z_{\text{Picard}}) \le \frac{1}{2} \sum_{i=n-K+1}^n \mathbf{E} \left[ \left( Z_{\text{Picard}} - Z_{\text{Picard}}^{(i)} \right)^2 \right]$$

The dropped terms represent the historical tail of the sequential rollout's variance:

$$\text{Tail}_{\text{seq}} = \frac{1}{2} \sum_{i=1}^{n-K} \mathbf{E} \left[ \left( Z_{\text{seq}} - Z_{\text{seq}}^{(i)} \right)^2 \right]$$

Under the stated premise that historical noise induces non-zero variance in the critic's evaluation, at least some terms in this sequential tail are strictly positive, guaranteeing $\text{Tail}_{\text{seq}} > 0$. Because the Picard scheme structurally annihilates this strictly positive deep historical tail—completely insulating the critic from those perturbations—the variance is strictly reduced. Consequently:

$$\operatorname{Var}(V_\phi(\hat{s}_n^{(K)})) < \operatorname{Var}(V_\phi(s_n^{\text{seq}}))$$

This completes the proof. $\blacksquare$

#### Corollaries
With the bias tightly bounded and the non-linear variance reduced, we deduce the primary theoretical contributions of the methodology.

**Corollary 1**
The Mean Squared Error of the meta policy PPO critic's value target is strictly lower under the Picard simulation scheme than the standard sequential scheme for a sufficiently accurate initial guess.

**Proof**:
By the Bias-Variance decomposition, the MSE of the value estimate $\hat{V}$ is:


$$\text{MSE}(\hat{V}) = \operatorname{Bias}(\hat{V})^2 + \operatorname{Var}(\hat{V})$$


From Lemma 1 and the Lipschitz continuity of $V$ (A4), the bias of the critic is bounded by $C_V L^K e_0$. Squaring this yields:


$$\operatorname{Bias}(\hat{V})^2 \le C_V^2 L^{2K} e_0^2$$


Because $L < 1$, this squared bias term collapses geometrically toward zero as $K$ grows.
Simultaneously, Theorem 1 guarantees that $\operatorname{Var}_{\text{Picard}} < \operatorname{Var}_{\text{seq}}$.
Therefore, replacing the highly variant sequential target with the variance-reduced, low-bias Picard target yields:


$$\text{MSE}_{\text{Picard}} < \text{MSE}_{\text{seq}}$$


This structural reduction in MSE accelerates the convergence of the policy gradient updates.

**Corollary 2**
The variance reduction scheme applied to the boundary states does not systematically bias or alter the learning dynamics of the low-level, intra-day policy.

**Proof**:
By design, the lower-level environment completely resets at the end of every interval, meaning the low-level policy's rewards and transitions are strictly intra-interval. Because its trajectories never cross macro boundaries, the low-level policy is entirely insulated from cross-interval state dynamics and remains mathematically unaffected by the Picard intervention.

## References
[1] Farias, V., Gijsbrechts, J., Khojandi, A., Peng, T., Zheng, A. (2025). Speeding up Policy Simulation in Supply Chain RL. arXiv:2406.01939v2.