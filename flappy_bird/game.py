import random

import pygame

pygame.init()
screen = pygame.display.set_mode((360, 640))
clock = pygame.time.Clock()
FPS = 60
running = True
pygame.display.set_caption("Flappy Bird")
background = pygame.image.load("assets/bg.png")
bird = pygame.image.load("assets/pngegg.png").convert_alpha()
pipe = pygame.image.load('assets/pngaaa.com-2304710.png').convert_alpha()

# Resize the images
background = pygame.transform.scale(background, (360, 640))
bird = pygame.transform.scale(bird, (50, 50))
pipe = pygame.transform.scale(pipe, (100, 400))

# initial bird position
bird_x = 50
bird_y = 50

# initial pipe position
pipe_x = 100
pipe_y = 50

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(background, (0, 0))
    screen.blit(bird, (bird_x, bird_y))
    screen.blit(pipe, (pipe_x, pipe_y))

    # move bird down every frame
    bird_y += 2

    # move pipe left every frame
    pipe_x -= 2

    # reset pipe position if it goes offscreen
    if pipe_x <= -100:
        pipe_x = 360

    # randomize pipe heights
    if pipe_x == 360:
        pipe_y = random.randint(0, 500)

    # check collisions
    if bird_y <= 0 or bird_y >= 590:
        running = False


    keys = pygame.key.get_pressed()

    # move bird up if space is pressed
    if keys[pygame.K_SPACE]:
        bird_y -= 5

    pygame.display.update()

    clock.tick(FPS)

pygame.quit()
