"""
Play Flappy Bird — choose Human or AI mode from the in-game menu.

Usage:
  python play.py
"""

import math
import pygame
import sys
import os
from flappy_env import FlappyEnv, SCREEN_W, SCREEN_H, GROUND_Y, COL_BG, COL_TEXT, COL_MUTED, COL_BIRD, COL_PIPE

# ─── Button helper ───────────────────────────────────────────────
def ensure_pygame_ready():
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

class Button:
    def __init__(self, x, y, w, h, text, color, hover_color, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False

    def draw(self, surf, font):
        col = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=10)
        pygame.draw.rect(surf, (255, 255, 255), self.rect, 2, border_radius=10)
        txt = font.render(self.text, True, self.text_color)
        surf.blit(txt, (
            self.rect.centerx - txt.get_width() // 2,
            self.rect.centery - txt.get_height() // 2
        ))

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


# ─── Menu screen ─────────────────────────────────────────────────

def show_menu(surf, clock):
    ensure_pygame_ready()
    title_font = pygame.font.SysFont('Arial', 44, bold=True)
    sub_font = pygame.font.SysFont('Arial', 18)
    btn_font = pygame.font.SysFont('Arial', 24, bold=True)

    btn_w, btn_h = 180, 55
    cx = SCREEN_W // 2

    human_btn = Button(cx - btn_w - 15, 320, btn_w, btn_h,
                       "Human", (50, 130, 200), (70, 160, 240))
    ai_btn = Button(cx + 15, 320, btn_w, btn_h,
                    "AI Bot", (200, 80, 50), (240, 100, 60))

    frame = 0

    while True:
        frame += 1
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if human_btn.clicked(mouse_pos):
                    return "human"
                if ai_btn.clicked(mouse_pos):
                    return "ai"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "human"
                if event.key == pygame.K_2:
                    return "ai"

        human_btn.update(mouse_pos)
        ai_btn.update(mouse_pos)

        # Draw background
        surf.fill(COL_BG)
        for y in range(0, SCREEN_H, 60):
            pygame.draw.line(surf, (20, 35, 60), (0, y), (SCREEN_W, y), 1)

        # Ground
        pygame.draw.rect(surf, (180, 140, 80),
                         pygame.Rect(0, GROUND_Y, SCREEN_W, SCREEN_H - GROUND_Y))
        pygame.draw.line(surf, (140, 100, 50), (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)

        # Decorative pipes
        for px in [60, 200, 340]:
            top_h = 120 + 40 * ((px // 60) % 3)
            pygame.draw.rect(surf, COL_PIPE, pygame.Rect(px, 0, 40, top_h), border_radius=4)
            pygame.draw.rect(surf, COL_PIPE,
                             pygame.Rect(px, top_h + 150, 40, GROUND_Y - top_h - 150),
                             border_radius=4)

        # Animated bird
        bird_y = 200 + int(8 * math.sin(frame * 0.08))
        bx = SCREEN_W // 2
        pygame.draw.circle(surf, COL_BIRD, (bx, bird_y), 20)
        pygame.draw.circle(surf, (200, 160, 20), (bx, bird_y), 20, 2)
        pygame.draw.circle(surf, (30, 30, 30), (bx + 9, bird_y - 5), 5)
        pygame.draw.circle(surf, (255, 255, 255), (bx + 10, bird_y - 6), 2)

        # Title
        title = title_font.render("FLAPPY BIRD", True, COL_TEXT)
        surf.blit(title, (cx - title.get_width() // 2, 100))
        rl_txt = sub_font.render("Reinforcement Learning Edition", True, COL_MUTED)
        surf.blit(rl_txt, (cx - rl_txt.get_width() // 2, 150))

        # Mode label
        choose_txt = sub_font.render("Choose your mode:", True, COL_TEXT)
        surf.blit(choose_txt, (cx - choose_txt.get_width() // 2, 290))

        # Buttons
        human_btn.draw(surf, btn_font)
        ai_btn.draw(surf, btn_font)

        # Keyboard hints
        hint1 = sub_font.render("Press 1", True, COL_MUTED)
        surf.blit(hint1, (human_btn.rect.centerx - hint1.get_width() // 2, 385))
        hint2 = sub_font.render("Press 2", True, COL_MUTED)
        surf.blit(hint2, (ai_btn.rect.centerx - hint2.get_width() // 2, 385))

        # Hover info
        if human_btn.hovered:
            info = sub_font.render("SPACE to flap — you control the bird", True, (100, 180, 255))
            surf.blit(info, (cx - info.get_width() // 2, 430))
        elif ai_btn.hovered:
            info = sub_font.render("Watch your trained DQN agent play", True, (255, 140, 100))
            surf.blit(info, (cx - info.get_width() // 2, 430))

        pygame.display.flip()
        clock.tick(30)


# ─── Game over screen ────────────────────────────────────────────

def show_game_over(surf, clock, score, mode):
    ensure_pygame_ready()
    big_font = pygame.font.SysFont('Arial', 40, bold=True)
    font = pygame.font.SysFont('Arial', 22, bold=True)
    btn_font = pygame.font.SysFont('Arial', 20, bold=True)
    hint_font = pygame.font.SysFont('Arial', 14)

    cx = SCREEN_W // 2
    btn_w, btn_h = 140, 45

    retry_btn = Button(cx - btn_w - 10, GROUND_Y // 2 + 50, btn_w, btn_h,
                       "Play Again", (50, 130, 200), (70, 160, 240))
    menu_btn = Button(cx + 10, GROUND_Y // 2 + 50, btn_w, btn_h,
                      "Menu", (100, 100, 100), (130, 130, 130))

    while True:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if retry_btn.clicked(mouse_pos):
                    return "retry"
                if menu_btn.clicked(mouse_pos):
                    return "menu"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return "retry"
                if event.key == pygame.K_ESCAPE:
                    return "menu"

        retry_btn.update(mouse_pos)
        menu_btn.update(mouse_pos)

        # Dark overlay
        overlay = pygame.Surface((SCREEN_W, GROUND_Y), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        surf.blit(overlay, (0, 0))

        # Text
        go_txt = big_font.render("GAME OVER", True, (255, 80, 80))
        surf.blit(go_txt, (cx - go_txt.get_width() // 2, GROUND_Y // 2 - 60))

        sc_txt = font.render(f"Score: {score}", True, COL_TEXT)
        surf.blit(sc_txt, (cx - sc_txt.get_width() // 2, GROUND_Y // 2 - 10))

        mode_label = "Human" if mode == "human" else "AI Bot"
        mode_txt = font.render(f"Mode: {mode_label}", True, COL_MUTED)
        surf.blit(mode_txt, (cx - mode_txt.get_width() // 2, GROUND_Y // 2 + 18))

        # Buttons
        retry_btn.draw(surf, btn_font)
        menu_btn.draw(surf, btn_font)

        h1 = hint_font.render("SPACE", True, COL_MUTED)
        surf.blit(h1, (retry_btn.rect.centerx - h1.get_width() // 2, retry_btn.rect.bottom + 5))
        h2 = hint_font.render("ESC", True, COL_MUTED)
        surf.blit(h2, (menu_btn.rect.centerx - h2.get_width() // 2, menu_btn.rect.bottom + 5))

        pygame.display.flip()
        clock.tick(30)


# ─── Human play ──────────────────────────────────────────────────

def play_human(env):
    state = env.reset()
    done = False

    while not done:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1

        state, reward, done = env.step(action)
        env.render(total_reward=env.score)

    return env.score


# ─── AI play ─────────────────────────────────────────────────────

def play_ai(env, agent):
    state = env.reset()
    done = False
    hint_font = pygame.font.SysFont('Arial', 16, bold=True)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "menu"

        action = agent.choose_action(state, greedy=True)
        state, reward, done = env.step(action)
        env.render(total_reward=env.score)
        hint = hint_font.render("ESC = Menu", True, COL_MUTED)
        env.surf.blit(hint, (SCREEN_W - hint.get_width() - 12, 12))
        pygame.display.flip()

    return env.score


# ─── Main loop ───────────────────────────────────────────────────

def main():
    ensure_pygame_ready()
    surf = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Flappy Bird — RL Edition")

    agent = None  # lazy load

    while True:
        mode = show_menu(surf, clock)

        # Load agent once if AI mode selected
        if mode == "ai" and agent is None:
            from dqn_agent import DQNAgent
            path = "flappy_dqn.pth"
            if not os.path.exists(path):
                print(f"No trained model at '{path}'. Train first with: python train.py")
                continue
            agent = DQNAgent()
            agent.load(path)

        # Game loop (retry stays in same mode)
        while True:
            env = FlappyEnv(render=True)

            if mode == "human":
                ensure_pygame_ready()
                font = pygame.font.SysFont('Arial', 22, bold=True)
                env.reset()
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            waiting = False
                    env.render()
                    txt = font.render("Press SPACE to start", True, COL_TEXT)
                    env.surf.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2, GROUND_Y // 2))
                    pygame.display.flip()
                    clock.tick(30)

                score = play_human(env)
            else:
                ai_result = play_ai(env, agent)
                if ai_result == "menu":
                    env.close()
                    break
                score = ai_result

            result = show_game_over(env.surf, clock, score, mode)
            env.close()

            if result == "menu":
                break  # back to mode selection
            # "retry" continues the loop in same mode


if __name__ == "__main__":
    main()
