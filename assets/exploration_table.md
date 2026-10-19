| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `dqn eps exp` | 300 | 250 | 0% | **0%** | -200.0 | 200 | 77 s |
| `dqn +count` | 300 | 59 | 100% | **100%** | -119.5 | 119 | 62 s |
| `dqn +rnd` | 300 | 250 | 100% | **100%** | -152.8 | 153 | 84 s |
| `dqn +per +n3` | 300 | 98 | 46% | **0%** | -200.0 | 200 | 104 s |
| `dqn +height shaping` | 300 | 59 | 28% | **90%** | -149.1 | 149 | 73 s |
| `q-learning +energy shaping` | 3000 | 38 | 100% | **100%** | -132.3 | 132 | 16 s |
