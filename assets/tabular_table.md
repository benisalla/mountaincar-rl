| run | episodes | first success (episode) | train success (last 50) | eval success | eval return | eval steps | time |
|---|---|---|---|---|---|---|---|
| `q-learning` | 3000 | 215 | 70% | **55%** | -161.8 | 162 | 14 s |
| `sarsa` | 3000 | 57 | 94% | **90%** | -158.0 | 158 | 11 s |
| `tile-coding (sarsa)` | 300 | 23 | 100% | **95%** | -136.8 | 137 | 4 s |
| `tile-coding (q)` | 300 | 27 | 100% | **100%** | -112.6 | 113 | 6 s |
| `cem` | 300 | 21 | 100% | **100%** | -113.2 | 113 | 2 s |
