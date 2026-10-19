| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `ppo` | 60 | never | 0% | **0%** | -0.0 | 999 | 59 s |
| `ppo +shaping` | 60 | 3 | 100% | **100%** | 94.7 | 150 | 20 s |
| `ddpg` | 30 | never | 0% | **0%** | -0.0 | 999 | 105 s |
| `ddpg +shaping` | 30 | 6 | 83% | **100%** | 90.2 | 101 | 6 s |
| `td3` | 30 | never | 0% | **0%** | -0.3 | 999 | 73 s |
| `td3 +shaping` | 30 | 6 | 83% | **100%** | 93.1 | 85 | 11 s |
| `sac` | 30 | never | 0% | **0%** | -0.0 | 999 | 120 s |
| `sac +shaping` | 30 | 6 | 83% | **100%** | 92.3 | 109 | 23 s |
| `cem` | 100 | 7 | 92% | **100%** | 96.9 | 290 | 1 s |
| `cem +shaping` | 100 | 7 | 92% | **100%** | 95.3 | 404 | 1 s |
