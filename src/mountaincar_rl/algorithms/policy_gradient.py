"""On-policy policy gradient: REINFORCE with a baseline, A2C and PPO (discrete or continuous).

The policy pi(a|s) is a Categorical over the 3 actions, or a Gaussian N(mu(s), sigma) over the
force (clipped to [-1, 1]); a separate critic V(s) estimates the value. All three maximise
E[ log pi(a|s) * A ], they only differ in how the advantage A is estimated and how often they update:

    REINFORCE: after each episode,  A_t = G_t - V(s_t),  G_t = r_t + gamma r_t+1 + ... (Monte Carlo)
    A2C:       every `a2c_steps` steps, A_t = n-step return bootstrapped with V - V(s_t)
    PPO:       every `rollout_steps` steps, GAE(lambda) advantages, then `ppo_epochs` passes of
               minibatch updates on the clipped ratio objective
               L = E[min(rho A, clip(rho, 1-eps, 1+eps) A)],  rho = pi(a|s) / pi_old(a|s)

    loss = -policy objective + 0.5 * (V(s) - target)^2 - entropy_coef * H[pi(.|s)]

Why they struggle here: without a single success, every return is exactly -200 (or close to 0 in
the continuous version, where doing nothing avoids the action cost). The advantage is then pure
noise, so the policy never gets a direction to improve in; in the continuous task the gradient
even pushes towards "do nothing", a well-known local optimum. Shaping gives them a dense signal.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from mountaincar_rl.algorithms.base import Agent
from mountaincar_rl.envs import normalize_obs
from mountaincar_rl.networks import mlp


class ActorCriticNet(nn.Module):
    def __init__(self, continuous: bool, hidden: int = 64, n_actions: int = 3):
        super().__init__()
        self.continuous = continuous
        self.actor = mlp(2, 1 if continuous else n_actions, hidden)
        self.critic = mlp(2, 1, hidden)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(1))

    def dist(self, x: torch.Tensor):
        out = self.actor(x)
        if self.continuous:
            return Normal(out, self.log_std.exp().expand_as(out))
        return Categorical(logits=out)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)


class PolicyGradientAgent(Agent):
    def __init__(self, cfg, n_actions: int = 3):
        super().__init__(cfg, n_actions)
        self.name, self.continuous = cfg.algo, cfg.continuous
        torch.manual_seed(cfg.seed)
        self.net = ActorCriticNet(cfg.continuous, cfg.hidden, n_actions)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self._clear()
        self._last_logp = 0.0

    def _clear(self) -> None:
        self.buf = {k: [] for k in ("obs", "actions", "logps", "rewards", "next_obs", "terminated", "ends")}

    # acting ---------------------------------------------------------------------------------
    def act(self, obs, explore: bool = True):
        x = torch.as_tensor(normalize_obs(obs)).unsqueeze(0)
        with torch.no_grad():
            d = self.net.dist(x)
            if explore:
                a = d.sample()
                self._last_logp = float(d.log_prob(a).sum())
            else:
                a = d.mean if self.continuous else d.probs.argmax(dim=-1)
        if self.continuous:
            self._last_raw = float(a.reshape(-1)[0])
            return np.clip(a.numpy().reshape(1), -1.0, 1.0).astype(np.float32)
        return int(a.item())

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        b = self.buf
        b["obs"].append(normalize_obs(obs))
        b["actions"].append(self._last_raw if self.continuous else int(action))
        b["logps"].append(self._last_logp)
        b["rewards"].append(float(reward))
        b["next_obs"].append(normalize_obs(next_obs))
        b["terminated"].append(float(terminated))
        b["ends"].append(float(terminated or truncated))
        n = len(b["rewards"])
        if self.cfg.algo == "a2c" and (n >= self.cfg.a2c_steps or terminated or truncated):
            self._update_actor_critic(lam=1.0, epochs=1, minibatch=None)
        elif self.cfg.algo == "ppo" and n >= self.cfg.rollout_steps:
            self._update_actor_critic(lam=self.cfg.gae_lambda, epochs=self.cfg.ppo_epochs, minibatch=64)

    def end_episode(self) -> None:
        if self.cfg.algo == "reinforce" and self.buf["rewards"]:
            self._update_reinforce()

    # learning -------------------------------------------------------------------------------
    def _tensors(self):
        b = self.buf
        obs = torch.as_tensor(np.array(b["obs"]))
        actions = torch.as_tensor(np.array(b["actions"]), dtype=torch.float32 if self.continuous else torch.long)
        if self.continuous:
            actions = actions.unsqueeze(-1)
        return obs, actions

    def _log_prob_entropy(self, obs, actions):
        d = self.net.dist(obs)
        logp = d.log_prob(actions)
        ent = d.entropy()
        if self.continuous:
            logp, ent = logp.sum(-1), ent.sum(-1)
        return logp, ent

    def _update_reinforce(self) -> None:
        obs, actions = self._tensors()
        rewards = self.buf["rewards"]
        returns, g = np.zeros(len(rewards), np.float32), 0.0
        for t in reversed(range(len(rewards))):           # G_t = r_t + gamma G_t+1
            g = rewards[t] + self.cfg.gamma * g
            returns[t] = g
        returns_t = torch.as_tensor(returns)
        values = self.net.value(obs)
        adv = returns_t - values.detach()                   # the baseline V(s) reduces variance only
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        logp, ent = self._log_prob_entropy(obs, actions)
        loss = -(logp * adv).mean() + 0.5 * ((values - returns_t) ** 2).mean() - self.cfg.entropy_coef * ent.mean()
        self._step(loss)
        self._clear()

    def _advantages(self, obs, lam: float):
        """GAE: delta_t = r_t + gamma (1 - term_t) V(s'_t) - V(s_t); A_t = delta_t + gamma lam (1 - end_t) A_t+1."""
        b = self.buf
        with torch.no_grad():
            v = self.net.value(obs).numpy()
            v_next = self.net.value(torch.as_tensor(np.array(b["next_obs"]))).numpy()
        rewards, term, ends = (np.array(b[k], np.float32) for k in ("rewards", "terminated", "ends"))
        adv, last = np.zeros_like(rewards), 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * (1.0 - term[t]) * v_next[t] - v[t]
            last = delta + self.cfg.gamma * lam * (1.0 - ends[t]) * last
            adv[t] = last
        return torch.as_tensor(adv), torch.as_tensor(adv + v)

    def _update_actor_critic(self, lam: float, epochs: int, minibatch: int | None) -> None:
        obs, actions = self._tensors()
        adv, returns = self._advantages(obs, lam)
        old_logp = torch.as_tensor(np.array(self.buf["logps"], np.float32))
        n = len(adv)
        for _ in range(epochs):
            order = torch.randperm(n) if minibatch else torch.arange(n)
            for start in range(0, n, minibatch or n):
                idx = order[start:start + (minibatch or n)]
                a = adv[idx]
                if minibatch and len(idx) > 1:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                logp, ent = self._log_prob_entropy(obs[idx], actions[idx])
                if self.cfg.algo == "ppo":
                    ratio = (logp - old_logp[idx]).exp()
                    clipped = ratio.clamp(1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
                    policy_loss = -torch.min(ratio * a, clipped * a).mean()
                else:
                    policy_loss = -(logp * a).mean()
                value_loss = ((self.net.value(obs[idx]) - returns[idx]) ** 2).mean()
                self._step(policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * ent.mean())
        self._clear()

    def _step(self, loss) -> None:
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()

    # inspection -----------------------------------------------------------------------------
    @torch.no_grad()
    def policy_grid(self, states):
        d = self.net.dist(torch.as_tensor(normalize_obs(states)))
        if self.continuous:
            return np.clip(d.mean.numpy()[:, 0], -1, 1)
        return d.probs.argmax(dim=-1).numpy()

    @torch.no_grad()
    def value_grid(self, states):
        return self.net.value(torch.as_tensor(normalize_obs(states))).numpy()

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state["net"])
