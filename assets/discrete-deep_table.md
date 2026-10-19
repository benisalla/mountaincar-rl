| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `dqn` | 300 | never | 0% | **0%** | -200.0 | 200 | 197 s |
| `dqn +shaping` | 300 | 65 | 100% | **95%** | -159.8 | 160 | 204 s |
| `double-dqn` | 300 | 165 | 4% | **80%** | -141.8 | 142 | 196 s |
| `double-dqn +shaping` | 300 | 65 | 100% | **100%** | -123.3 | 123 | 143 s |
| `dueling-dqn` | 300 | never | 0% | **0%** | -200.0 | 200 | 274 s |
| `dueling-dqn +shaping` | 300 | 65 | 100% | **100%** | -138.3 | 138 | 149 s |
| `reinforce` | 500 | never | 0% | **0%** | -200.0 | 200 | 45 s |
| `reinforce +shaping` | 500 | 238 | 24% | **95%** | -129.2 | 129 | 34 s |
| `a2c` | 500 | never | 0% | **0%** | -200.0 | 200 | 83 s |
| `a2c +shaping` | 500 | 371 | 100% | **100%** | -141.1 | 141 | 80 s |
| `ppo` | 500 | 198 | 100% | **100%** | -142.2 | 142 | 76 s |
| `ppo +shaping` | 500 | 135 | 100% | **100%** | -140.0 | 140 | 203 s |
