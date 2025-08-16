# Optimal Control for CartPole Probelm
Learning and optimizing controllers for the classic cart–pole task, with two approaches:
- LQR (Continuous & Discrete): classic linear optimal control around the upright.
- iLQR: nonlinear iterative LQR with a receding horizon.


## Setup
To set up the environment and install all required packages:
```
uv sync
```
> [!TIP]
> Make sure uv is installed (`pip install uv`).


## Run the Linear Quadratic Regulator (LQR)
```
uv run run.py -c <config: clqr, dlqr, ilqr, pinn>
```


## TODO
- [ ] Implement initial state
- [ ] Refactor interface
