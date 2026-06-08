"""Parallel day execution for Picard fixed-point iteration.

Days within a Picard iteration are independent once the solver pre-samples
the BM trajectory, day seeds, and meta-policy noise. This module dispatches
them to a fork-based ``multiprocessing.Pool``.

The fiddly bit is gradient flow. PyTorch's autograd graph doesn't survive
pickling, so shipping ``SavedAction(log_prob, value)`` tensors back would
silently turn them into leaf tensors and ``backward()`` in main would produce
zeros. Instead, each worker runs the math of ``A2C.training_step`` on its own
day's buffer — but as a *sum* rather than a *mean* — calls ``backward()``
locally, and ships per-parameter detached CPU grad tensors. Main sums the
per-day grads (linearity of differentiation: ∇Σℓ_d = Σ∇ℓ_d) and divides by
the total number of transitions to recover the same value the sequential
``mean()`` would have produced. The result is identical to sequential
``training_step`` to within floating-point reduction order.

Two side-channel concerns get handled here too:
  * **wandb**: workers can't call ``wandb.log`` (no run context in children).
    ``day_runner.py`` appends to ``ctx.day_logs`` instead; the wrapper drains.
  * **PicardSolver state**: workers don't carry one. ``_WorkerShim`` quacks
    like the solver, captures a ``WorkerDayCapture``, and the wrapper feeds
    each capture to ``central_picard.merge_worker_capture`` in i_day order.
"""

import multiprocessing as mp
import random as _random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.algos.picard.solver import WorkerDayCapture, WorkerDayPayload
from src.misc.day_runner import run_day
from src.misc.utils import DailyStatsAccumulator


# Module-level state populated in the parent before pool fork. Workers
# inherit it via copy-on-write (fork start method).
_WORKER_STATE: dict = {}


# ── worker-side shim that quacks like PicardSolver ────────────────────────────

class _WorkerShim:
    """Per-day stand-in for ``PicardSolver`` used inside worker processes.

    ``prepare_day`` mirrors the central solver's side effects from the
    payload; ``record_day`` captures the day's outcome for the wrapper to
    replay onto the central solver via ``merge_worker_capture``.
    """

    def __init__(self, payload: WorkerDayPayload):
        self._payload = payload
        self.captured: Optional[WorkerDayCapture] = None

    def prepare_day(self, i_day: int, env) -> dict:
        env.brand_momentum = dict(self._payload.bm_in)
        np.random.seed(self._payload.day_seed)
        env._shuffle_rng = _random.Random(self._payload.day_seed)
        torch.manual_seed(self._payload.day_seed)
        return dict(self._payload.meta_alpha)

    def record_day(self, i_day, env, accumulator, meta_reward, wandb_metrics) -> None:
        accumulator.momentum_snapshot = dict(env.brand_momentum)
        self.captured = WorkerDayCapture(
            bm_out=dict(env.brand_momentum),
            meta_obs={
                a: accumulator.daily_state(
                    a, i_day + 1, self._payload.num_days, self._payload.reward_scalar
                )
                for a in self._payload.agents
            },
            meta_reward=dict(meta_reward),
            wandb_metrics=dict(wandb_metrics),
        )


# ── dataclasses that cross the process boundary ───────────────────────────────

@dataclass
class GradPacket:
    """Per-agent, per-day partial gradients + bookkeeping for aggregation."""
    n_steps: int
    actor_grads: dict       # {param_name: detached cpu tensor}
    critic_grads: dict
    p_loss_sum: float       # Σ policy_losses (for actor_loss metric)
    v_loss_sum: float       # Σ value_losses
    advantages: list        # per-step advantages (for adv mean/std metric)


@dataclass
class WorkerResult:
    i_day: int
    capture: WorkerDayCapture
    ctx_after: SimpleNamespace        # per-day deltas to merge into central ctx
    grad_packets: dict                # {agent_id: GradPacket} non-fix agents only
    day_logs: list                    # wandb dicts queued by run_day


# ── gradient helpers ──────────────────────────────────────────────────────────

def _compute_partial_gradients(agent) -> GradPacket:
    """Run A2C.training_step's math on one day's buffer, as a SUM.

    Sequential training_step does ``stack(policy_losses).mean().backward()``
    over the union of all days' transitions. Here we do
    ``stack(policy_losses_d).sum().backward()`` per day; the wrapper sums
    those gradients across days and divides by total transitions to recover
    the mean-loss gradient.

    Returns per-parameter detached CPU grad tensors plus the loss-sum scalars
    (for logging actor_loss / critic_loss as the eventual mean).
    """
    agent.actor.zero_grad()
    agent.critic.zero_grad()

    saved_actions = agent.saved_actions
    rewards = agent.rewards
    n = len(rewards)

    if n == 0:
        return GradPacket(0, {}, {}, 0.0, 0.0, [])

    # Returns with the same per-day terminal logic as training_step. With a
    # single day in this buffer the "boundary" is just the last step, where R
    # would start at 0 anyway — equivalent to the simpler per-day computation.
    returns = []
    R = 0.0
    for r in rewards[::-1]:
        R = r + agent.gamma * R
        returns.insert(0, R)
    returns_t = torch.tensor(returns, device=agent.device) / agent.reward_scale

    advantages = [
        float(R_t.item()) - float(value.item())
        for (log_prob, value), R_t in zip(saved_actions, returns_t)
    ]

    policy_losses = []
    value_losses = []
    for (log_prob, value), R_t, adv in zip(saved_actions, returns_t, advantages):
        policy_losses.append(-log_prob * adv)
        value_losses.append(F.smooth_l1_loss(value, R_t.detach().unsqueeze(0)))

    # Critical: SUM (not mean). Wrapper divides by total_steps to recover mean.
    p_loss_sum = torch.stack(policy_losses).sum()
    v_loss_sum = torch.stack(value_losses).sum()
    p_loss_sum.backward()
    v_loss_sum.backward()

    actor_grads = {
        name: p.grad.detach().cpu().clone()
        for name, p in agent.actor.named_parameters() if p.grad is not None
    }
    critic_grads = {
        name: p.grad.detach().cpu().clone()
        for name, p in agent.critic.named_parameters() if p.grad is not None
    }

    return GradPacket(
        n_steps=n,
        actor_grads=actor_grads,
        critic_grads=critic_grads,
        p_loss_sum=float(p_loss_sum.item()),
        v_loss_sum=float(v_loss_sum.item()),
        advantages=advantages,
    )


def _zero_metrics() -> dict:
    return {
        'actor_grad_norm': 0.0, 'critic_grad_norm': 0.0,
        'actor_loss': 0.0, 'critic_loss': 0.0,
        'advantage_mean': 0.0, 'advantage_std': 0.0,
    }


def _apply_aggregated_gradients_one_agent(agent, packets, update_actor: bool) -> dict:
    """Sum per-day grads, divide by total transitions, clip, step.

    Returns the same metrics dict shape as ``A2C.training_step``.
    """
    nonempty = [p for p in packets if p.n_steps > 0]
    if not nonempty:
        return _zero_metrics()
    total_steps = sum(p.n_steps for p in nonempty)

    agent.actor.zero_grad()
    agent.critic.zero_grad()

    template = nonempty[0]
    # param.grad = (Σ_d worker_grad_d) / total_steps  ↔  ∇ of sequential mean loss
    for name, param in agent.actor.named_parameters():
        if name in template.actor_grads:
            summed = sum(p.actor_grads[name] for p in nonempty if name in p.actor_grads)
            param.grad = (summed / total_steps).to(param.device)
    for name, param in agent.critic.named_parameters():
        if name in template.critic_grads:
            summed = sum(p.critic_grads[name] for p in nonempty if name in p.critic_grads)
            param.grad = (summed / total_steps).to(param.device)

    if update_actor:
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            agent.actor.parameters(), agent.actor_clip
        )
        agent.optimizers['a_optimizer'].step()
        actor_grad_norm_val = float(actor_grad_norm.item())
    else:
        actor_grad_norm_val = 0.0

    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        agent.critic.parameters(), agent.critic_clip
    )
    agent.optimizers['c_optimizer'].step()

    # Sequential training_step clears these at the end. Central agent's
    # buffers are never populated in parallel mode, but clear for parity.
    del agent.rewards[:]
    del agent.saved_actions[:]

    all_advs = [a for p in nonempty for a in p.advantages]
    return {
        'actor_grad_norm': actor_grad_norm_val,
        'critic_grad_norm': float(critic_grad_norm.item()),
        'actor_loss': sum(p.p_loss_sum for p in nonempty) / total_steps,
        'critic_loss': sum(p.v_loss_sum for p in nonempty) / total_steps,
        'advantage_mean': float(np.mean(all_advs)) if all_advs else 0.0,
        'advantage_std': float(np.std(all_advs)) if all_advs else 0.0,
    }


# ── ctx (de)construction ──────────────────────────────────────────────────────

def _build_worker_ctx(env, payload: WorkerDayPayload, i_episode: int):
    """Construct a fresh ctx for one day's ``run_day`` call.

    Accumulators are zeroed so the worker's *final* ctx values **are** the
    per-day deltas the wrapper will merge into the central ctx.
    """
    args = _WORKER_STATE['args']
    model_agents = _WORKER_STATE['model_agents']
    meta_policies = _WORKER_STATE['meta_policies']
    solveRebFlow = _WORKER_STATE['solveRebFlow']

    accumulator = DailyStatsAccumulator()
    shim = _WorkerShim(payload)

    ctx = SimpleNamespace(
        env=env, args=args, model_agents=model_agents,
        meta_policies=meta_policies, accumulator=accumulator,
        picard_solver=shim, solveRebFlow=solveRebFlow,
        i_episode=i_episode,
        # Initial obs is overwritten on the first match_step inside run_day.
        obs=None,
        action_rl={a: [0.0] * env.nregion for a in [0, 1]},
        meta_obs={a: np.zeros(7, dtype=np.float32) for a in [0, 1]},
        meta_multipliers={a: 1.0 for a in [0, 1]},
        # Zeroed accumulators — final values are per-day deltas.
        episode_reward={0: 0, 1: 0},
        episode_served_demand={0: 0, 1: 0},
        episode_unserved_demand={0: 0, 1: 0},
        episode_rebalancing_cost={0: 0, 1: 0},
        episode_total_revenue={0: 0, 1: 0},
        episode_total_operating_cost={0: 0, 1: 0},
        episode_waiting={0: 0, 1: 0},
        episode_rejected_demand=0,
        episode_total_demand=0,
        episode_rejection_rates=[],
        episode_true_profit={0: 0, 1: 0},
        episode_adjusted_profit={0: 0, 1: 0},
        episode_unprofitable_trips={0: 0, 1: 0},
        actions_price={0: [], 1: []},
        actions_effective_price={0: [], 1: []},
        meta_shaping_term={0: [], 1: []},
        episode_logprobs={0: [], 1: []},
        day_logs=[],
        env_pre_reset=True,  # day_runner: skip its inner env.reset_day (we did it)
    )
    if env.mode == 0:
        ctx.actions_concentration_dirichlet = {0: [], 1: []}
        ctx.episode_min_concentration_dirichlet = {0: float('inf'), 1: float('inf')}
        ctx.episode_max_concentration_dirichlet = {0: float('-inf'), 1: float('-inf')}
    elif env.mode == 1:
        ctx.actions_concentration_alpha = {0: [], 1: []}
        ctx.actions_concentration_beta = {0: [], 1: []}
        ctx.episode_min_concentration_alpha = {0: float('inf'), 1: float('inf')}
        ctx.episode_max_concentration_alpha = {0: float('-inf'), 1: float('-inf')}
        ctx.episode_min_concentration_beta = {0: float('inf'), 1: float('inf')}
        ctx.episode_max_concentration_beta = {0: float('-inf'), 1: float('-inf')}
    else:  # mode 2
        ctx.actions_concentration_alpha = {0: [], 1: []}
        ctx.actions_concentration_beta = {0: [], 1: []}
        ctx.actions_concentration_dirichlet = {0: [], 1: []}
        ctx.episode_min_concentration_alpha = {0: float('inf'), 1: float('inf')}
        ctx.episode_max_concentration_alpha = {0: float('-inf'), 1: float('-inf')}
        ctx.episode_min_concentration_beta = {0: float('inf'), 1: float('inf')}
        ctx.episode_max_concentration_beta = {0: float('-inf'), 1: float('-inf')}
        ctx.episode_min_concentration_dirichlet = {0: float('inf'), 1: float('inf')}
        ctx.episode_max_concentration_dirichlet = {0: float('-inf'), 1: float('-inf')}
    return ctx, shim


def _strip_ctx(ctx) -> SimpleNamespace:
    """Strip non-pickleable references (env, model_agents, picard_solver, …)
    so only the per-day deltas cross the process boundary.
    """
    fields = [
        'episode_reward', 'episode_served_demand', 'episode_unserved_demand',
        'episode_rebalancing_cost', 'episode_total_revenue',
        'episode_total_operating_cost', 'episode_waiting',
        'episode_rejected_demand', 'episode_total_demand',
        'episode_rejection_rates',
        'episode_true_profit', 'episode_adjusted_profit', 'episode_unprofitable_trips',
        'actions_price', 'actions_effective_price', 'meta_shaping_term',
        'episode_logprobs',
    ]
    out = SimpleNamespace(**{f: getattr(ctx, f) for f in fields})
    for f in (
        'actions_concentration_alpha', 'actions_concentration_beta',
        'actions_concentration_dirichlet',
        'episode_min_concentration_alpha', 'episode_max_concentration_alpha',
        'episode_min_concentration_beta', 'episode_max_concentration_beta',
        'episode_min_concentration_dirichlet', 'episode_max_concentration_dirichlet',
    ):
        setattr(out, f, getattr(ctx, f, None))
    return out


# ── worker entry point ────────────────────────────────────────────────────────

def _worker(task):
    """Run one day in a worker process.

    Seeds RNG with ``payload.day_seed`` *before* ``env.reset()`` so the
    demand realisation is deterministic across Picard iterations within an
    episode (otherwise the fixed point wouldn't exist — two iterations
    would see different demand for the same i_day).
    """
    i_episode, payload = task
    state = _WORKER_STATE
    env = state['env']
    args = state['args']
    model_agents = state['model_agents']

    # Deterministic env state for this (episode, day).
    np.random.seed(payload.day_seed)
    env.reset()

    # Worker copies of the agents accumulate only this day's transitions.
    for a in model_agents:
        if a != args.fix_agent:
            model_agents[a].rewards = []
            model_agents[a].saved_actions = []

    ctx, shim = _build_worker_ctx(env, payload, i_episode)
    run_day(payload.i_day, ctx)

    grad_packets = {}
    if args.mode not in [3, 4]:
        for a in [0, 1]:
            if a != args.fix_agent:
                grad_packets[a] = _compute_partial_gradients(model_agents[a])

    return WorkerResult(
        i_day=payload.i_day,
        capture=shim.captured,
        ctx_after=_strip_ctx(ctx),
        grad_packets=grad_packets,
        day_logs=ctx.day_logs,
    )


# ── pool lifecycle ────────────────────────────────────────────────────────────

def _init_worker() -> None:
    """Per-worker init. Single thread so N workers don't oversubscribe."""
    torch.set_num_threads(1)
    import os as _os
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_var] = "1"


def set_worker_state(env, model_agents, args, meta_policies, solveRebFlow) -> None:
    """Populate the module-level worker state. MUST be called *before*
    ``make_pool`` so the fork inherits these references.
    """
    import os as _os
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_var] = "1"
    global _WORKER_STATE
    _WORKER_STATE = {
        'env': env, 'model_agents': model_agents, 'args': args,
        'meta_policies': meta_policies, 'solveRebFlow': solveRebFlow,
    }


def make_pool(num_workers: int):
    """Create a fork-based worker pool. Call after ``set_worker_state``."""
    ctx = mp.get_context('fork')
    return ctx.Pool(num_workers, initializer=_init_worker)


def _apply_state_dicts(state_dicts) -> None:
    """Worker function: load fresh weights into the inherited model_agents."""
    ma = _WORKER_STATE['model_agents']
    for a, sd in state_dicts.items():
        ma[a].actor.load_state_dict(sd['actor'])
        ma[a].critic.load_state_dict(sd['critic'])


def broadcast_state_dicts(pool, num_workers: int, model_agents: dict, fix_agent: int) -> None:
    """Push central model_agents weights to every worker.

    Workers diverge from main after each ``optimizer.step()``, so this must
    run once per episode before any day work. ``chunksize=1`` with one task
    per worker guarantees every worker applies the update.
    """
    state_dicts = {
        a: {
            'actor': {k: v.detach().cpu() for k, v in model_agents[a].actor.state_dict().items()},
            'critic': {k: v.detach().cpu() for k, v in model_agents[a].critic.state_dict().items()},
        }
        for a in model_agents if a != fix_agent
    }
    pool.map(_apply_state_dicts, [state_dicts] * num_workers, chunksize=1)


# ── orchestrator (public) ─────────────────────────────────────────────────────

def run_days_parallel(ctx, picard_solver, pool) -> list:
    """Run one Picard iteration's days in parallel and merge results.

    Side effects on ``ctx``:
      - per-day deltas summed into ``ctx.episode_*`` accumulators
      - lists (``actions_*``, ``episode_logprobs``, ``episode_rejection_rates``,
        mode-specific concentration trackers) extended in day order
      - min/max reductions applied to concentration min/max dicts
      - ``ctx.day_logs`` extended in day order

    Side effects on ``picard_solver``:
      - ``merge_worker_capture`` called per day in order, populating
        ``_day_results`` exactly as ``record_day`` would have

    Returns the list of ``WorkerResult`` (sorted by i_day) so the caller
    keeps the final Picard iteration's ``grad_packets`` for the
    post-convergence aggregation.
    """
    num_days = picard_solver.num_days
    env_mode = ctx.env.mode

    payloads = [picard_solver.make_worker_payload(i) for i in range(num_days)]
    tasks = [(ctx.i_episode, p) for p in payloads]
    results = pool.map(_worker, tasks, chunksize=1)
    results.sort(key=lambda r: r.i_day)   # defensive; pool.map preserves order

    for r in results:
        picard_solver.merge_worker_capture(r.i_day, r.capture)
        _merge_ctx_delta(ctx, r.ctx_after, env_mode)
        ctx.day_logs.extend(r.day_logs)

    return results


def apply_aggregated_gradients(results, model_agents: dict, args, update_actor: bool) -> dict:
    """Apply the final Picard iteration's worker grads to central agents.

    Returns ``grad_norms`` in the same shape as the sequential
    ``training_step`` so the existing wandb logging is unchanged.
    """
    grad_norms = {}
    if results is None or args.mode in [3, 4]:
        for a in [0, 1]:
            grad_norms[a] = _zero_metrics()
        return grad_norms

    for a in [0, 1]:
        if a == args.fix_agent:
            grad_norms[a] = _zero_metrics()
            # Sequential training_step clears fix_agent buffers; do the same
            # on the central agent for parity (workers cleared their copies).
            del model_agents[a].rewards[:]
            del model_agents[a].saved_actions[:]
        else:
            packets = [r.grad_packets[a] for r in results if a in r.grad_packets]
            grad_norms[a] = _apply_aggregated_gradients_one_agent(
                model_agents[a], packets, update_actor
            )
    return grad_norms


# ── per-day delta merge (in-place into central ctx) ───────────────────────────

def _merge_ctx_delta(ctx, delta, env_mode: int) -> None:
    for a in [0, 1]:
        ctx.episode_reward[a] += delta.episode_reward[a]
        ctx.episode_served_demand[a] += delta.episode_served_demand[a]
        ctx.episode_unserved_demand[a] += delta.episode_unserved_demand[a]
        ctx.episode_rebalancing_cost[a] += delta.episode_rebalancing_cost[a]
        ctx.episode_total_revenue[a] += delta.episode_total_revenue[a]
        ctx.episode_total_operating_cost[a] += delta.episode_total_operating_cost[a]
        ctx.episode_waiting[a] += delta.episode_waiting[a]
        ctx.episode_true_profit[a] += delta.episode_true_profit[a]
        ctx.episode_adjusted_profit[a] += delta.episode_adjusted_profit[a]
        ctx.episode_unprofitable_trips[a] += delta.episode_unprofitable_trips[a]
        ctx.actions_price[a].extend(delta.actions_price[a])
        ctx.actions_effective_price[a].extend(delta.actions_effective_price[a])
        ctx.meta_shaping_term[a].extend(delta.meta_shaping_term[a])
        ctx.episode_logprobs[a].extend(delta.episode_logprobs[a])
    ctx.episode_rejected_demand += delta.episode_rejected_demand
    ctx.episode_total_demand += delta.episode_total_demand
    ctx.episode_rejection_rates.extend(delta.episode_rejection_rates)

    if env_mode == 0:
        for a in [0, 1]:
            ctx.actions_concentration_dirichlet[a].extend(delta.actions_concentration_dirichlet[a])
            ctx.episode_min_concentration_dirichlet[a] = min(
                ctx.episode_min_concentration_dirichlet[a], delta.episode_min_concentration_dirichlet[a])
            ctx.episode_max_concentration_dirichlet[a] = max(
                ctx.episode_max_concentration_dirichlet[a], delta.episode_max_concentration_dirichlet[a])
    elif env_mode == 1:
        for a in [0, 1]:
            ctx.actions_concentration_alpha[a].extend(delta.actions_concentration_alpha[a])
            ctx.actions_concentration_beta[a].extend(delta.actions_concentration_beta[a])
            ctx.episode_min_concentration_alpha[a] = min(
                ctx.episode_min_concentration_alpha[a], delta.episode_min_concentration_alpha[a])
            ctx.episode_max_concentration_alpha[a] = max(
                ctx.episode_max_concentration_alpha[a], delta.episode_max_concentration_alpha[a])
            ctx.episode_min_concentration_beta[a] = min(
                ctx.episode_min_concentration_beta[a], delta.episode_min_concentration_beta[a])
            ctx.episode_max_concentration_beta[a] = max(
                ctx.episode_max_concentration_beta[a], delta.episode_max_concentration_beta[a])
    else:  # mode 2
        for a in [0, 1]:
            ctx.actions_concentration_alpha[a].extend(delta.actions_concentration_alpha[a])
            ctx.actions_concentration_beta[a].extend(delta.actions_concentration_beta[a])
            ctx.actions_concentration_dirichlet[a].extend(delta.actions_concentration_dirichlet[a])
            ctx.episode_min_concentration_alpha[a] = min(
                ctx.episode_min_concentration_alpha[a], delta.episode_min_concentration_alpha[a])
            ctx.episode_max_concentration_alpha[a] = max(
                ctx.episode_max_concentration_alpha[a], delta.episode_max_concentration_alpha[a])
            ctx.episode_min_concentration_beta[a] = min(
                ctx.episode_min_concentration_beta[a], delta.episode_min_concentration_beta[a])
            ctx.episode_max_concentration_beta[a] = max(
                ctx.episode_max_concentration_beta[a], delta.episode_max_concentration_beta[a])
            ctx.episode_min_concentration_dirichlet[a] = min(
                ctx.episode_min_concentration_dirichlet[a], delta.episode_min_concentration_dirichlet[a])
            ctx.episode_max_concentration_dirichlet[a] = max(
                ctx.episode_max_concentration_dirichlet[a], delta.episode_max_concentration_dirichlet[a])
