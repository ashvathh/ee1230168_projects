import pygame
import numpy as np 
import random

SCREEN_W = 400
SCREEN_H = 600
GRAVITY = 0.5
FLAP_STR = -9
MAX_VEL = 10
PIPE_SPEED = 3
PIPE_GAP = 150
PIPE_FREQ = 90 # frames between pipe spawns
BIRD_X = 80
BIRD_R = 16 # bird radius

# colours
COL_BG = (10,20,40)
COL_BIRD = (255,220,50)
COL_PIPE = (80,180,80)
COL_PIPE_EDG = (50,140,50)
COL_TEXT = (240,240,240)
COL_MUTED = (120,120,120)
COL_GROUND = (180,140,80)

GROUND_Y = SCREEN_H - 60

# --- PIPE CLASS ---

class Pipe:
    W = 55

    def __init__(self, x):
        self.x = x # horizontal position of the pipe
        gap_center = random.randint(180, GROUND_Y-180)
        self.top_y = gap_center - PIPE_GAP//2
        self.bot_y = gap_center + PIPE_GAP//2
        self.passed = False # has the bird passed this pipe?

    def update(self):
        self.x -= PIPE_SPEED # move pipe left

    def off_screen(self):
        return self.x + self.W < 0 # check if pipe is off the left edge
    
    def collides(self, bird_x, bird_y, bird_r):
        # Vertical collision bands
        in_x = bird_x + bird_r > self.x and bird_x - bird_r < self.x + self.W
        hit_top = in_x and bird_y - bird_r < self.top_y
        hit_bot = in_x and bird_y + bird_r > self.bot_y

        return in_x and (hit_top or hit_bot)
    
    def draw(self, surf):
        # Draw top pipe
        top_rect = pygame.Rect(self.x, 0, self.W, self.top_y)
        pygame.draw.rect(surf, COL_PIPE, top_rect, border_radius=4)
        pygame.draw.rect(surf, COL_PIPE_EDG, top_rect, 2, border_radius=4)

        # cap on top pipe
        cap = pygame.Rect(self.x - 4, self.top_y - 18, self.W + 8, 18)
        pygame.draw.rect(surf, COL_PIPE, cap, border_radius=4)
        pygame.draw.rect(surf, COL_PIPE_EDG, cap, 2, border_radius=4)

        # bottom pipe
        bot_rect = pygame.Rect(self.x, self.bot_y, self.W, GROUND_Y - self.bot_y)
        pygame.draw.rect(surf, COL_PIPE, bot_rect, border_radius=4)
        pygame.draw.rect(surf, COL_PIPE_EDG, bot_rect, 2, border_radius=4)
        # cap on bottom pipe
        cap2 = pygame.Rect(self.x - 4, self.bot_y, self.W + 8, 18)
        pygame.draw.rect(surf, COL_PIPE, cap2, border_radius=4)
        pygame.draw.rect(surf, COL_PIPE_EDG, cap2, 2, border_radius=4)

# Environment class to manage game state and logic

class FlappyEnv:
    def __init__(self, render=False):
        self.render_mode = render
        self.surf = None
        self.clock = None
        self.font = None
        
        if self.render_mode:
            pygame.init()
            self.surf = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18, bold=True)
            pygame.display.set_caption("Flappy Bird - RL Agent")

    def reset(self):
        self.bird_y = SCREEN_H//2
        self.bird_vel = 0
        self.pipes = [Pipe(SCREEN_W + 100)]
        self.frame = 0
        self.score = 0
        self.alive = True
        return self._get_state()
    
    def step(self, action):
        assert self.alive, "Episode has ended. Call reset() to start a new episode."

        # action: 0 = do nothing, 1 = flap
        if action == 1:
            self.bird_vel = FLAP_STR
        
        # physics update
        self.bird_vel = min(self.bird_vel + GRAVITY, MAX_VEL)
        self.bird_y += self.bird_vel
        self.frame += 1

        # spawn new pipes
        if self.frame % PIPE_FREQ == 0:
            self.pipes.append(Pipe(SCREEN_W + 10))

        # update pipes
        for p in self.pipes:
            p.update()
        
        self.pipes = [p for p in self.pipes if not p.off_screen()]

        # ===== REVISED REWARD STRUCTURE =====
        
        # 1) Small survival reward every frame — teaches "don't die"
        reward = 0

        # 2) Pipe clearance reward (big positive signal)
        for p in self.pipes:
            if not p.passed and p.x + p.W < BIRD_X:
                p.passed = True
                self.score += 1
                reward += 10.0

        # 3) Gap-centering reward — encourages positioning near the gap
        next_pipe = self._next_pipe()
        if next_pipe:
            gap_center = (next_pipe.top_y + next_pipe.bot_y) / 2
            dist = abs(self.bird_y - gap_center)
            max_dist = SCREEN_H / 2
            if dist < gap_center - PIPE_GAP//2 or dist > gap_center + PIPE_GAP//2:
                reward -= 0.01 * (dist / max_dist)  # small penalty for being far from gap
            else:
                reward += 0.02 * (1.0 - dist / max_dist)

        # 4) Collision / out-of-bounds check
        done = False

        if self.bird_y - BIRD_R <= 0 or self.bird_y + BIRD_R >= GROUND_Y:
            done = True
            reward = -5.0   # reduced from -15; Huber loss handles the rest

        for p in self.pipes:
            if p.collides(BIRD_X, self.bird_y, BIRD_R):
                done = True
                reward = -5.0  # reduced from -10
                break
        
        self.alive = not done
        return self._get_state(), reward, done
    

    # State Vector

    def _get_state(self):
        pipe = self._next_pipe()

        if pipe:
            pipe_dist_x = (pipe.x - BIRD_X) / SCREEN_W
            pipe_top_y = pipe.top_y / SCREEN_H
            pipe_bot_y = pipe.bot_y / SCREEN_H
            gap_center_y = (pipe.top_y + pipe.bot_y) / 2
            dist_to_gap = (self.bird_y - gap_center_y) / SCREEN_H
        else:
            pipe_dist_x = 1.0
            pipe_top_y = 0.5
            pipe_bot_y = 0.7
            dist_to_gap = 0.0

        return np.array([
            self.bird_y / SCREEN_H,
            self.bird_vel / MAX_VEL,
            pipe_dist_x,
            pipe_top_y,
            pipe_bot_y,
            dist_to_gap
        ], dtype=np.float32)
    
    def _next_pipe(self):
        ahead = [p for p in self.pipes if p.x + p.W > BIRD_X - 10]
        return min(ahead, key=lambda p: p.x) if ahead else None
    
    # Rendering
    def render(self, episode=None, total_reward=None):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.surf.fill(COL_BG)

        for y in range(0, SCREEN_H, 60):
            pygame.draw.line(self.surf, (20, 35, 60), (0, y), (SCREEN_W, y), 1)

        for p in self.pipes:
            p.draw(self.surf)

        pygame.draw.rect(self.surf, COL_GROUND,
                         pygame.Rect(0, GROUND_Y, SCREEN_W, SCREEN_H - GROUND_Y))
        pygame.draw.line(self.surf, (140, 100, 50), (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)

        bx, by = int(BIRD_X), int(self.bird_y)
        pygame.draw.circle(self.surf, COL_BIRD, (bx, by), BIRD_R)
        pygame.draw.circle(self.surf, (200, 160, 20), (bx, by), BIRD_R, 2)
        pygame.draw.circle(self.surf, (30, 30, 30), (bx + 7, by - 4), 4)
        pygame.draw.circle(self.surf, (255, 255, 255), (bx + 8, by - 5), 2)

        vel_disp = max(-BIRD_R, min(BIRD_R, int(self.bird_vel * 1.5)))
        pygame.draw.line(self.surf, (255, 180, 0),
                         (bx - 8, by), (bx - 8, by + vel_disp), 2)

        pipe = self._next_pipe()
        if pipe:
            gap_c = int((pipe.top_y + pipe.bot_y) / 2)
            pygame.draw.line(self.surf, (80, 80, 120),
                             (pipe.x, gap_c), (pipe.x + pipe.W, gap_c), 1)

        score_t = self.font.render(f"Score  {self.score}", True, COL_TEXT)
        self.surf.blit(score_t, (14, 14))

        if episode is not None:
            ep_t = self.font.render(f"Episode  {episode}", True, COL_MUTED)
            self.surf.blit(ep_t, (14, 38))

        if total_reward is not None:
            r_t = self.font.render(f"Reward  {total_reward:.1f}", True, COL_MUTED)
            self.surf.blit(r_t, (14, 62))

        state = self._get_state()
        labels = ["bird_y", "bird_vel", "pipe_dist", "top_y", "bot_y", "gap_dist"]

        for i, (lbl, val) in enumerate(zip(labels, state)):
            s = pygame.font.SysFont('Courier New', 11).render(
                f"{lbl}: {val:+.3f}", True, (80, 100, 140))
            self.surf.blit(s, (SCREEN_W - 160, 14 + i * 16))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_mode:
            self.surf = None
            self.clock = None
            self.font = None
