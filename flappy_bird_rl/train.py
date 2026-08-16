import pygame
import numpy as np 
import os
from flappy_env import FlappyEnv
from dqn_agent import DQNAgent

# CONFIG
EPISODES = 5000
RENDER_EVERY = 200
SAVE_EVERY = 100
TRAIN_EVERY = 2       # train every 2 steps (was 4) — more frequent updates
PRINT_EVERY = 20

SAVE_PATH = "flappy_dqn.pth"

# Logging

class TrainingLog:
    def __init__(self):
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_lengths = []
        self.losses = []
        self.epsilons = []

    def push(self, reward, score, length, loss, epsilon):
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_lengths.append(length)
        self.losses.append(loss if loss is not None else 0)
        self.epsilons.append(epsilon)

    def recent(self, n=20):
        return {
            "reward": np.mean(self.episode_rewards[-n:]),
            "score": np.mean(self.episode_scores[-n:]),
            "length": np.mean(self.episode_lengths[-n:]),
            "loss": np.mean(self.losses[-n:]),
            "best_score": max(self.episode_scores) if self.episode_scores else 0,
        }

# Single episode

def run_episode(env, agent, render=False, episode_num=None, train=True):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    losses = []

    while not done:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, None, None, None
        
        action = agent.choose_action(state, greedy=not train)
        next_state, reward, done = env.step(action)

        if train:
            agent.store(state, action, reward, next_state, float(done))

            # Train every TRAIN_EVERY steps
            if agent.steps_done % TRAIN_EVERY == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            # SOFT sync target net every step (tau=0.005 makes this smooth)
            agent.sync_target()
        
        total_reward += reward
        steps += 1
        state = next_state

        if render:
            env.render(episode=episode_num, total_reward=total_reward)
        
    avg_loss = np.mean(losses) if losses else None
    return total_reward, env.score, steps, avg_loss

# Training loop

def train():
    agent = DQNAgent(
        state_dim       = 6,
        action_dim      = 2,
        lr              = 5e-4,       # slightly lower LR for stability
        gamma           = 0.99,
        epsilon         = 1.0,
        epsilon_min     = 0.02,       # lower floor — almost fully greedy at end
        epsilon_decay   = 0.997,      # slower decay — reaches min around ep 2000+
        batch_size      = 64,
        buffer_capacity = 100000,     # 10x larger buffer
        warmup_steps    = 2000,
    )

    # load checkpoint if exists
    if os.path.exists(SAVE_PATH):
        ans = input(f"Found saved model at '{SAVE_PATH}'. Load and resume? (y/n): ")
        if ans.strip().lower() == 'y':
            agent.load(SAVE_PATH)
    
    log = TrainingLog()

    print("\nStarting training...")
    print(f"  Episodes:      {EPISODES}")
    print(f"  Render every:  {RENDER_EVERY}")
    print(f"  Save every:    {SAVE_EVERY}")
    print(f"  Buffer cap:    {agent.buffer.buffer.maxlen}")
    print(f"  Warmup steps:  {agent.warmup_steps}")
    print(f"  Tau (soft):    {agent.tau}")
    print(f"  Epsilon decay: {agent.epsilon_decay}\n")
    
    for episode in range(1, EPISODES + 1):
        should_render = (episode % RENDER_EVERY == 0)

        env = FlappyEnv(render=should_render)

        result = run_episode(
            env, agent,
            render=should_render,
            episode_num=episode,
            train=True
        )

        env.close()

        if result[0] is None:
            print("Training interrupted by user.")
            break

        total_reward, score, length, avg_loss = result

        agent.decay_epsilon()

        log.push(total_reward, score, length, avg_loss, agent.epsilon)

        if episode % PRINT_EVERY == 0:
            r = log.recent(PRINT_EVERY)
            warmup = len(agent.buffer) < agent.warmup_steps
            status = "WARMUP" if warmup else "TRAINING"
            print(
                f"ep {episode:>5} | "
                f"reward {r['reward']:>7.2f} | "
                f"score {r['score']:>5.2f} | "
                f"steps {r['length']:>6.0f} | "
                f"loss {r['loss']:>8.5f} | "
                f"ε {agent.epsilon:.3f} | "
                f"buf {len(agent.buffer):>6} | "
                f"best {r['best_score']} | "
                f"{status}"
            )

        if episode % SAVE_EVERY == 0:
            agent.save(SAVE_PATH)
            print(f"Saved checkpoint → {SAVE_PATH}")
        
    agent.save(SAVE_PATH)
    print(f"Final model saved → {SAVE_PATH}")
    plot(log)
    return agent, log

# Plotting
def plot(log):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Flappy Bird DQN Training', fontsize=14)

    def smooth(data, window=20):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    eps = range(1, len(log.episode_rewards) + 1)

    axes[0,0].plot(eps, log.episode_rewards, alpha=0.3, color='steelblue')
    axes[0,0].plot(range(len(smooth(log.episode_rewards))),
                   smooth(log.episode_rewards), color='steelblue')
    axes[0,0].set_title('Total reward per episode')
    axes[0,0].set_xlabel('Episode')

    axes[0,1].plot(eps, log.episode_scores, alpha=0.3, color='coral')
    axes[0,1].plot(range(len(smooth(log.episode_scores))),
                   smooth(log.episode_scores), color='coral')
    axes[0,1].set_title('Pipes cleared per episode')
    axes[0,1].set_xlabel('Episode')

    axes[1,0].plot(eps, log.losses, alpha=0.3, color='gray')
    axes[1,0].plot(range(len(smooth(log.losses))),
                   smooth(log.losses), color='gray')
    axes[1,0].set_title('Training loss')
    axes[1,0].set_xlabel('Episode')

    axes[1,1].plot(eps, log.epsilons, color='orange')
    axes[1,1].set_title('Epsilon decay')
    axes[1,1].set_xlabel('Episode')

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.show()


# Watching trained agent
def watch(path=SAVE_PATH):
    if not os.path.exists(path):
        print(f"No saved model found at '{path}'. Train first.")
        return

    agent = DQNAgent()
    agent.load(path)

    print("\nWatching trained agent. Close window to quit.\n")

    episode = 0
    while True:
        episode += 1
        env = FlappyEnv(render=True)
        result = run_episode(env, agent, render=True,
                             episode_num=episode, train=False)
        env.close()

        if result[0] is None:
            break

        total_reward, score, steps, _ = result
        print(f"Episode {episode} — score: {score}  reward: {total_reward:.2f}  steps: {steps}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'watch':
        watch()
    else:
        train()
