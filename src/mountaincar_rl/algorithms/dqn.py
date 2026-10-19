"""DQN, Double DQN and Dueling DQN, with optional prioritized replay and n-step returns.

One gradient step per environment step on a minibatch from the replay buffer:

    DQN:        y = r + gamma^n (1 - done) max_a' Q_target(s', a')
    Double DQN: y = r + gamma^n (1 - done) Q_target(s', argmax_a' Q_online(s', a'))
                (the online net picks the action, the target net scores it: less over-estimation)
    Dueling:    Double DQN target with the dueling network Q = V + A - mean(A)
    loss = mean_i  w_i * Huber(Q_online(s_i, a_i) - y_i)     (w_i = 1 without prioritized replay)

The target net is a copy of the online net refreshed every `target_update` steps, so the
regression target does not move at every step. Exploration is epsilon-greedy with a schedule.

The lesson on MountainCar: a neural net initialised near 0 is NOT reliably optimistic (unlike the
table), and epsilon-greedy dithering almost never builds enough energy to reach the flag in 200
steps. Until one success lands in the buffer every target says "-1 forever" and nothing is learnt.
Shaping, an intrinsic bonus or a lucky episode is needed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mountaincar_rl.algorithms.base import Agent
from mountaincar_rl.envs import normalize_obs
from mountaincar_rl.networks import DuelingQNet, QNet
from mountaincar_rl.replay import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer


class DQNAgent(Agent):
    def __init__(self, cfg, n_actions: int = 3):
        super().__init__(cfg, n_actions)
        self.name = cfg.algo
        torch.manual_seed(cfg.seed)
        net_cls = DuelingQNet if cfg.algo == "dueling-dqn" else QNet
        self.q = net_cls(cfg.hidden, n_actions)
        self.q_target = net_cls(cfg.hidden, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        buf_cls = PrioritizedReplayBuffer if cfg.prioritized else ReplayBuffer
        self.buffer = buf_cls(cfg.buffer_size, seed=cfg.seed)
        self.nstep = NStepAccumulator(cfg.n_step, cfg.gamma)
        self.double = cfg.algo in ("double-dqn", "dueling-dqn")
        self.steps = 0

    @torch.no_grad()
    def q_values(self, states) -> np.ndarray:
        x = torch.as_tensor(normalize_obs(np.atleast_2d(states)))
        return self.q(x).numpy()

    def act(self, obs, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)[0]))

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        for s, a, r, s2, done, discount in self.nstep.push(obs, action, reward, next_obs, terminated, truncated):
            self.buffer.add(normalize_obs(s), a, r, normalize_obs(s2), done, discount)
        self.steps += 1
        if len(self.buffer) >= self.cfg.warmup_steps:
            self.update()
        if self.steps % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def update(self) -> float:
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.beta = 0.4 + 0.6 * self.progress          # anneal importance correction to 1
        b = self.buffer.sample(self.cfg.batch_size)
        obs = torch.as_tensor(b["obs"])
        actions = torch.as_tensor(b["actions"][:, 0]).long()
        rewards, dones = torch.as_tensor(b["rewards"]), torch.as_tensor(b["dones"])
        next_obs, discounts = torch.as_tensor(b["next_obs"]), torch.as_tensor(b["discounts"])
        with torch.no_grad():
            if self.double:
                best = self.q(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.q_target(next_obs).gather(1, best).squeeze(1)
            else:
                next_q = self.q_target(next_obs).max(dim=1).values
            target = rewards + discounts * (1.0 - dones) * next_q
        q_sa = self.q(obs).gather(1, actions[:, None]).squeeze(1)
        td = q_sa - target
        loss = (torch.as_tensor(b["weights"]) * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.buffer.update_priorities(b["idx"], td.detach().numpy())
        return float(loss.detach())

    def policy_grid(self, states):
        return self.q_values(states).argmax(axis=1)

    def value_grid(self, states):
        return self.q_values(states).max(axis=1)

    def state_dict(self):
        return {"q": self.q.state_dict()}

    def load_state_dict(self, state):
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q"])
