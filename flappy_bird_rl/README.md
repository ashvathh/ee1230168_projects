# Flappy Bird — Deep Q-Network RL Agent

A Deep Q-Network (DQN) agent that learns to play Flappy Bird through reinforcement learning, built from scratch using PyTorch and Pygame. No RL libraries — just the Bellman equation, a replay buffer, and a target network.

After training, the agent plays indefinitely without dying.

---

## How it works

### State representation

The agent never sees raw pixels. Each frame is represented as a 6-value vector:

```
[ bird_y, bird_velocity, pipe_dist_x, pipe_top_y, pipe_bot_y, dist_to_gap ]
```

All values are normalized to roughly `[-1, 1]`. This trains in minutes on CPU rather than hours on GPU.

### Action space

Two actions — flap (`1`) or do nothing (`0`). The network outputs a Q-value for each.

### Network architecture

```
Input:   6 neurons   (state vector)
Hidden:  128 neurons + ReLU
Hidden:  128 neurons + ReLU
Hidden:  64 neurons  + ReLU
Output:  2 neurons   (Q-value per action, no activation)
```

### Reward structure

```
+10.0                             clearing a pipe
+0.02 * (1.0 - dist / max_dist)   being aligned with the gap center (dense shaping reward)
+0.01 * (dist / max_dist)         small penalty for being far from gap
-5.0                              hitting a pipe, ceiling, or ground
```

The gap-centering reward is critical — without it the agent receives almost no signal for the first hundred episodes and learns nothing. It gives the agent a gradient to follow on every frame instead of waiting for the sparse pipe-clear signal.

### Key DQN components

**Replay buffer** — stores the last 100,000 transitions. Training samples random batches of 64, breaking the correlation between consecutive frames that would otherwise destabilize the network.

**Target network** — a second frozen copy of the network used to compute Bellman targets. Synced every 10 episodes. Prevents the moving-target instability where the network chases its own shifting predictions.

**Double DQN** — the policy network selects the best next action, the target network evaluates it. Reduces Q-value overestimation that causes training instability.

**Soft target updates** — target network weights are blended smoothly (`τ=0.005`) rather than copied hard, giving more stable Bellman targets.

**Huber loss** — less sensitive to large TD errors from death transitions than MSE, preventing single bad episodes from dominating the gradient.

### The Bellman update

```
Q(s, a)  ←  Q(s, a)  +  α × [ r  +  γ × Q_target(s', argmax Q_policy(s'))  −  Q(s, a) ]
```

`γ = 0.99` — the agent strongly values future rewards, connecting the flap decision at frame 40 to the pipe-clear reward at frame 55.

### Why inference outperforms training

During training, `epsilon=0.02` means 1 in 50 frames is a random action. One wrong random flap at the wrong moment kills the episode — artificially suppressing training scores. The policy is better than the training graphs suggest.

During inference, `epsilon=0.0` — pure policy, zero random actions. The true quality of the learned policy reveals itself. This is the **exploration-exploitation gap**, a standard phenomenon in RL.

---

## Project structure

```
flappy_bird/
├── flappy_env.py          — game environment (physics, pipes, collision, reward, rendering)
├── dqn_agent.py           — DQN network, replay buffer, Double DQN training step
├── train.py               — training loop with periodic live rendering + checkpointing
├── play.py                — play as human or watch the trained agent via pygame UI
├── flappy_dqn.pth         — saved model weights (generated after training)
└── flappy_dqn_buffer.pkl  — saved replay buffer for resuming training
```

---

## Setup

```bash
pip install pygame torch numpy matplotlib
```

---

## Usage

### Train the agent

```bash
python train.py
```

- Trains for 5,000 episodes
- Renders a live game every 200 episodes so you can watch the agent improve
- Saves a checkpoint every 100 episodes to `flappy_dqn.pth`
- If a checkpoint exists, prompts to resume training from where you left off
- Plots training curves at the end

### Watch the trained agent or play yourself

```bash
python play.py
```

Opens a menu to choose between:

- **Human** — control the bird with `SPACE`
- **AI Bot** — watch the trained agent play

---

## Training progress

The agent goes through distinct phases:

| Episodes | Behaviour |
|---|---|
| 0 – 500 | Dies immediately, pure random exploration, buffer filling |
| 500 – 1500 | Starts surviving a few seconds, no pipe awareness yet |
| 1500 – 2500 | Clears first pipes occasionally, learns gap alignment |
| 2500 – 4000 | Consistently clears 2–5 pipes, policy stabilising |
| 4000 – 5000 | Greedy policy plays near-indefinitely |

---

## Key concepts implemented

| Concept | Where |
|---|---|
| Bellman equation | `dqn_agent.py` — `train_step()` |
| Experience replay | `dqn_agent.py` — `ReplayBuffer` class |
| Target network | `dqn_agent.py` — `policy_net` vs `target_net` |
| Double DQN | `train_step()` — policy selects, target evaluates |
| Soft target update | `sync_target()` — `τ=0.005` blend |
| Epsilon-greedy | `choose_action()` — decays `1.0 → 0.02` |
| Reward shaping | `flappy_env.py` — gap-centering reward |
| Gradient clipping | `train_step()` — `clip_grad_norm_(..., 1.0)` |
| Huber loss | `train_step()` — `SmoothL1Loss` |
| Resume training | `save()` / `load()` — weights + optimizer + buffer |

---

## Design decisions

**Why not raw pixels?** Pixel input requires a convolutional network and hours of GPU training. A handcrafted 6-value state vector captures everything the agent needs and trains on CPU in under an hour.

**Why Double DQN?** Standard DQN overestimates Q-values because the same network both selects and evaluates actions. Double DQN separates these roles — policy net picks the action, target net scores it — reducing overestimation bias and stabilising training.

**Why Huber loss over MSE?** Death transitions carry reward `-5.0` while survival frames carry `+0.02`. MSE squares the large death error, causing it to dominate every update. Huber loss clips large errors linearly, letting the network learn from all transitions proportionally.

**Why gap-centering reward?** Without it, the only non-zero reward in the first hundred episodes is `-5.0` on death. The agent has no gradient during survival — it cannot tell if it is doing well until it dies. Gap-centering provides a small reward every frame for being near where it needs to be, dramatically accelerating early learning.
