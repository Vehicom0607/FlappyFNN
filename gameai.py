import pygame
import random
import numpy as np

pygame.init()
font = pygame.font.Font("arial.ttf", 25)

GRAVITY = 1
JUMP_HEIGHT = 18
PIPE_SPEED = 5
BIRD_COLOR = (255, 250, 205)
PIPE_COLOR = (183, 255, 47)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
OFFSET = 100  # How offset from the center on the x axis

# Bird images
bird_frames = []
bird_frames.append(pygame.transform.scale2x(pygame.image.load('sprites/yellowbird-downflap.png')))
bird_frames.append(pygame.transform.scale2x(pygame.image.load('sprites/yellowbird-midflap.png')))
bird_frames.append(pygame.transform.scale2x(pygame.image.load('sprites/yellowbird-upflap.png')))

pipe_surface = pygame.image.load('sprites/pipe-green.png')
pipe_surface = pygame.transform.scale2x(pipe_surface)

pipe_heights = [600]
PIPE_COOLDOWN = 72
ANIMATION_MORE_LIKE_ANIMESH_COOLDOWN = 12

# Audio
flap_sound = pygame.mixer.Sound("audio/wing.wav")
death_sound = pygame.mixer.Sound("audio/hit.wav")
score_sound = pygame.mixer.Sound("audio/point.wav")


class FlappyBird:
    def __init__(self, width=576, height=1024):
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()

        self.bird_indx = 0
        self.bird_surface = bird_frames[self.bird_indx]
        self.bird_rect = self.bird_surface.get_rect(center=(self.width / 2 - OFFSET, self.height / 2))
        self.delta_y = 0
        self.score = 0
        self.framecount = 0
        self.scuff = 0

        self.pipe_list = []

    def reset(self):
        self.bird_rect.centery = self.height / 2
        self.delta_y = 10  # Add a little bit at the start for a human player to adjust

        self.pipe_list = []
        self.framecount = 0
        self.score = 0
        self.pipe_list.extend(self.spawn_pipe())
        self.scuff = 0

    def spawn_pipe(self):
        random_pipe_pos = random.choice(pipe_heights)
        bottom_pipe = pipe_surface.get_rect(midtop=(700, random_pipe_pos))
        top_pipe = pipe_surface.get_rect(midbottom=(700, random_pipe_pos - 300))
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        for pipe in pipes:
            pipe.centerx -= PIPE_SPEED
        return pipes

    def remove_pipes(self, pipes):
        for pipe in pipes:
            if pipe.centerx == -600:
                pipes.remove(pipe)
        return pipes

    def draw_pipes(self, pipes):
        for pipe in pipes:
            if pipe.bottom >= self.height:
                self.screen.blit(pipe_surface, pipe)
            else:
                flipped_pipe = pygame.transform.flip(pipe_surface, False, True)  # Do a little flip upside down
                self.screen.blit(flipped_pipe, pipe)

    def rotate_bird(self, bird):
        return pygame.transform.rotozoom(bird, -1 * self.delta_y * 3, 1)

    def bird_animation(self):
        new_bird = bird_frames[self.bird_indx]
        new_bird_rect = new_bird.get_rect(center=(self.width / 2 - OFFSET, self.bird_rect.centery))
        return new_bird, new_bird_rect

    def check_game_over(self, pipes):
        for pipe in pipes:
            if self.bird_rect.colliderect(pipe):
                return True
        if self.bird_rect.top <= -100 or self.bird_rect.bottom >= self.height:
            return True
        return False

    def display_score(self):
        score_surface = font.render("SCORE: " + str(self.score), True, WHITE)
        score_rect = score_surface.get_rect(center=(100, 50))
        self.screen.blit(score_surface, score_rect)

    # Made into it's own function to make programing the AI easier
    def flap(self):
        flap_sound.play()
        self.delta_y = -JUMP_HEIGHT

    def play_step(self, action):
        self.framecount += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # AI's Action
        if action == 1:
            self.flap()
        # Fill screen
        self.screen.fill(BLACK)

        # Animation
        if self.framecount % ANIMATION_MORE_LIKE_ANIMESH_COOLDOWN == 0:
            if self.bird_indx > 2:
                self.bird_indx += 1
            else:
                self.bird_indx = 0
            self.bird_surface, self.bird_rect = self.bird_animation()

        # Movement
        self.delta_y += GRAVITY
        self.bird_rect.y += self.delta_y
        self.screen.blit(self.rotate_bird(self.bird_surface), self.bird_rect)

        # Pipes
        if self.framecount % PIPE_COOLDOWN == 0:
            self.pipe_list.extend(self.spawn_pipe())

        self.pipe_list = self.move_pipes(self.pipe_list)
        self.pipe_list = self.remove_pipes(self.pipe_list)
        self.draw_pipes(self.pipe_list)

        # Check Game Over and Update Score
        reward = 1  # Survive = 1, Game Over = - 10
        game_over = self.check_game_over(self.pipe_list)
        if game_over:
            death_sound.play()
            reward = -10
        if (self.framecount - (self.width / 2 - OFFSET) // PIPE_SPEED) % PIPE_COOLDOWN == 0:
            print(self.scuff, self.score)
            if self.scuff < 1:
                self.scuff += 1
            else:
                self.score += 1
            reward = 10
            score_sound.play()
        self.display_score()

        pygame.display.update()

        return reward, game_over, self.score

    def get_state(self):
        centerx = 0
        top = 0
        if len(self.pipe_list) != 0:
            pipe = self.pipe_list[0]
            pipe_num = 0
            while pipe.centerx <= self.bird_rect.centerx:
                pipe_num += 2
                pipe = self.pipe_list[pipe_num]
                centerx = pipe.centerx
                top = pipe.top
        else:
            centerx = self.width
            top = self.height / 2
        state = [
            self.delta_y,
            self.bird_rect.centery,
            centerx,
            top
        ]
        return np.array(state, dtype=int)
