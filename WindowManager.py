import pygame

# pre-define some colours
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
RED = (255, 0, 0)

def drawWindow(window, mode, traffic):
    draw_intersection(window, mode)
    draw_cars(window, traffic)
    pygame.display.update()

def draw_traffic_lights(window, x, y, mode):
    light_color = GREEN if mode == 0 else RED

    # traffic light background
    pygame.draw.rect(window, BLACK, (x, y, 20, 60))

    # draws light green or red depending on mode
    pygame.draw.circle(window, light_color, (x + 10, y + 15), 10)

def draw_intersection(window, mode):
    window.fill(DARK_GREEN)

    # draws the roads
    pygame.draw.rect(window, GRAY, (300, 0, 200, 600))  # Vertical road
    pygame.draw.rect(window, GRAY, (0, 225, 800, 200))  # Horizontal road

    # draws the traffic lights
    draw_traffic_lights(window, 280, 250, mode - 0)  # Left
    draw_traffic_lights(window, 500, 350, mode - 1)  # Right
    draw_traffic_lights(window, 450, 200, mode - 2)  # Top
    draw_traffic_lights(window, 350, 400, mode - 3)  # Bottom

def draw_cars(window, traffic):
    carsLeft = 0
    carsTop = 0
    carsRight = 0
    carsBottom = 0

    for car in traffic.left:
        carsLeft += 1
    for car in traffic.top:
        carsTop += 1
    for car in traffic.right:
        carsRight += 1
    for car in traffic.bottom:
        carsBottom += 1

    # for car in carsOnBoard:
    #     if car.loc == [1]:
    #         carsW = carsW + 1
    #     elif car.loc == [2]:
    #         carsN = carsN + 1
    #     elif car.loc == [3]:
    #         carsE = carsE + 1
    #     elif car.loc == [4]:
    #         carsS = carsS + 1

    font = pygame.font.SysFont(None, 24)
    text = font.render(f'Cars: {carsLeft}', False, WHITE)
    window.blit(text, (175, 275))

    font = pygame.font.SysFont(None, 24)
    text = font.render(f'Cars: {carsTop}', False, WHITE)
    window.blit(text, (425, 150))

    font = pygame.font.SysFont(None, 24)
    text = font.render(f'Cars: {carsRight}', False, WHITE)
    window.blit(text, (550, 375))

    font = pygame.font.SysFont(None, 24)
    text = font.render(f'Cars: {carsBottom}', False, WHITE)
    window.blit(text, (325, 475))