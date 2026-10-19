"""Off-policy actor-critic for the continuous car: DDPG, TD3 and SAC.

All three keep a replay buffer, a critic Q(s, a) trained by TD regression and an actor trained
to output actions the critic likes. They differ in three small places:

    DDPG: deterministic actor mu(s) = tanh(...), Gaussian action noise for exploration
          y = r + gamma (1 - done) Q'(s', mu'(s'))
          actor loss = -Q(s, mu(s))
    TD3:  DDPG + two critics with y using min(Q1', Q2')  (fights over-estimation)
          + target policy smoothing  a' = clip(mu'(s') + clip(noise), -1, 1)
          + actor and targets updated every `policy_delay` critic steps
    SAC:  stochastic actor a = tanh(u), u ~ N(mu(s), sigma(s)); maximises reward + alpha * entropy
          y = r + gamma (1 - done) (min(Q1', Q2')(s', a') - alpha log pi(a'|s')),  a' ~ pi(.|s')
          actor loss = alpha log pi(a|s) - min(Q1, Q2)(s, a);  alpha tuned to a target entropy

Targets Q' and mu' follow the online nets slowly: theta' <- tau theta + (1 - tau) theta'.

On MountainCarContinuous the trap is the action cost (-0.1 a^2 per step): before the flag has
ever been reached, the best thing the critic knows is "push as little as possible". DDPG/TD3 need
strong noise to stumble on the flag once; SAC's entropy bonus keeps it pushing randomly longer.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mountaincar_rl.algorithms.base import Agent
from mountaincar_rl.envs import normalize_obs
from mountaincar_rl.networks import mlp
from mountaincar_rl.replay import ReplayBuffer

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Critic(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = mlp(3, 1, hidden)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class GaussianActor(nn.Module):
    """SAC actor: tanh-squashed Gaussian, log-prob corrected for the tanh."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = mlp(2, 2, hidden)

    def forward(self, s, deterministic: bool = False):
        mu, log_std = self.net(s).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # log N(u; mu, std) - log(1 - tanh(u)^2)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


class OffPolicyActorCritic(Agent):
    continuous = True

    def __init__(self, cfg, n_actions: int = 1):
        super().__init__(cfg, n_actions)
        self.name = cfg.algo
        torch.manual_seed(cfg.seed)
        h = cfg.hidden
        if cfg.algo == "sac":
            self.actor = GaussianActor(h)
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -1.0
        else:
            self.actor = nn.Sequential(mlp(2, 1, h), nn.Tanh())
        self.critics = nn.ModuleList([Critic(h)] + ([Critic(h)] if cfg.algo in ("td3", "sac") else []))
        self.actor_target = copy.deepcopy(self.actor)
        self.critics_target = copy.deepcopy(self.critics)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, seed=cfg.seed)
        self.steps = 0

    @torch.no_grad()
    def _policy(self, states, deterministic=True) -> np.ndarray:
        x = torch.as_tensor(normalize_obs(np.atleast_2d(states)))
        if self.cfg.algo == "sac":
            return self.actor(x, deterministic=deterministic)[0].numpy()
        return self.actor(x).numpy()

    def act(self, obs, explore: bool = True):
        if explore and self.steps < self.cfg.warmup_steps:
            return self.rng.uniform(-1, 1, size=1).astype(np.float32)   # random warm-up
        if self.cfg.algo == "sac":
            a = self._policy(obs, deterministic=not explore)[0]
        else:
            a = self._policy(obs)[0]
            if explore:
                a = a + self.cfg.action_noise * self.rng.standard_normal(1)
        return np.clip(a, -1, 1).astype(np.float32)

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        self.buffer.add(normalize_obs(obs), action, reward, normalize_obs(next_obs), float(terminated))
        self.steps += 1
        if len(self.buffer) >= self.cfg.warmup_steps:
            self.update()

    def _soft_update(self, net, target) -> None:
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    def update(self) -> None:
        c = self.cfg
        b = self.buffer.sample(c.batch_size)
        s, a = torch.as_tensor(b["obs"]), torch.as_tensor(b["actions"])
        r, d, s2 = torch.as_tensor(b["rewards"]), torch.as_tensor(b["dones"]), torch.as_tensor(b["next_obs"])
        with torch.no_grad():
            if c.algo == "sac":
                a2, logp2 = self.actor(s2)
                q2 = torch.min(*[q(s2, a2) for q in self.critics_target]) - self.log_alpha.exp() * logp2
            elif c.algo == "td3":
                noise = (0.2 * torch.randn_like(a)).clamp(-0.5, 0.5)
                a2 = (self.actor_target(s2) + noise).clamp(-1, 1)
                q2 = torch.min(*[q(s2, a2) for q in self.critics_target])
            else:
                q2 = self.critics_target[0](s2, self.actor_target(s2))
            y = r + c.gamma * (1 - d) * q2
        critic_loss = sum(F.mse_loss(q(s, a), y) for q in self.critics)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if c.algo == "td3" and self.steps % c.policy_delay != 0:
            return
        if c.algo == "sac":
            a_new, logp = self.actor(s)
            q_new = torch.min(*[q(s, a_new) for q in self.critics])
            actor_loss = (self.log_alpha.exp().detach() * logp - q_new).mean()
        else:
            actor_loss = -self.critics[0](s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        if c.algo == "sac":   # temperature: raise alpha when entropy < target, lower it otherwise
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        self._soft_update(self.critics, self.critics_target)
        self._soft_update(self.actor, self.actor_target)

    def policy_grid(self, states):
        return self._policy(states)[:, 0]

    @torch.no_grad()
    def value_grid(self, states):
        x = torch.as_tensor(normalize_obs(states))
        a = torch.as_tensor(self._policy(states))
        return torch.min(torch.stack([q(x, a) for q in self.critics]), dim=0).values.numpy()

    def state_dict(self):
        state = {"actor": self.actor.state_dict(), "critics": self.critics.state_dict()}
        if self.cfg.algo == "sac":
            state["log_alpha"] = self.log_alpha.detach().clone()
        return state

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.actor_target.load_state_dict(state["actor"])
        self.critics_target.load_state_dict(state["critics"])
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"])
