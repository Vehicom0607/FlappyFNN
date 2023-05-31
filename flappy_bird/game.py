import pygame

pygame.init()
screen = pygame.display.set_mode((360, 640))
clock = pygame.time.Clock()
FPS = 60
running = True
pygame.display.set_caption("Flappy Bird")
background = pygame.image.load("assets/bg.png")
bird = pygame.image.load("pngegg.png")
pipe = pygame.image.load('assets/pngaaa.com-®2304710.png')

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(background, (0, 0))
    screen.draw(bird, (100, 100))

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        print("Space bar pressed")

    pygame.display.update()
