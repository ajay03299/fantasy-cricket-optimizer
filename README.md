# Fantasy-Cricket-Optimizer

A project to build a fantasy cricket decision engine using historical cricket data, fantasy point mapping, machine learning, and optimization.

## Current Progress

### Completed
- Cleaned player-match dataset created from ball-by-ball data
- `fantasy_points_v0.py` built and tested
- IPL-style fantasy proxy score generated for each player-match row

### Fantasy Points V0 includes
- Runs scored
- Boundary bonus
- Six bonus
- 50-run bonus
- 100-run bonus
- Strike-rate penalty
- Wicket points
- 4-wicket bonus
- 5-wicket bonus
- Economy bonus
- Catch points
- Stumping points
- Run-out proxy points

### Not yet included in V0
- Announced lineup bonus
- Duck penalty
- Maiden over bonus
- Direct vs assisted run-out split
- Role-aware strike-rate logic
- Opponent and role enrichment

## Next Steps
- Validate fantasy points output on sample matches
- Enrich dataset with missing fantasy-critical fields
- Build `fantasy_points_v1`
- Create model-ready feature table
- Build optimization layer for team selection