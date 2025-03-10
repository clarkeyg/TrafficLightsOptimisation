import pygame

# Pre-defined colors
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
RED = (255, 0, 0)

def draw_window(window, mode, traffic):
    draw_intersection(window, mode)
    draw_cars(window, traffic)
    pygame.display.update()

def draw_traffic_lights(window, x, y, is_green):
    light_color = GREEN if is_green else RED
    pygame.draw.rect(window, BLACK, (x, y, 20, 60))
    pygame.draw.circle(window, light_color, (x + 10, y + 15), 10)

def draw_intersection(window, mode):
    window.fill(DARK_GREEN)
    pygame.draw.rect(window, GRAY, (300, 0, 200, 600))  # Vertical road
    pygame.draw.rect(window, GRAY, (0, 225, 800, 200))  # Horizontal road
    draw_traffic_lights(window, 280, 250, mode == 0)  # Left
    draw_traffic_lights(window, 500, 350, mode == 1)  # Right
    draw_traffic_lights(window, 450, 200, mode == 2)  # Top
    draw_traffic_lights(window, 350, 400, mode == 3)  # Bottom

def draw_car(window, x, y, color):
    pygame.draw.rect(window, color, (x, y, 20, 10))

def draw_cars(window, traffic):
    # Left lane: cars approaching from x=0 to x=300, y=250
    for i, car in enumerate(traffic.left):
        x = 300 - i * 25  # First car closest to intersection
        y = 250
        if x >= 0:  # Ensure cars stay on screen
            draw_car(window, x, y, car.colour)

    # Right lane: cars from x=800 to x=500, y=400
    for i, car in enumerate(traffic.right):
        x = 500 + i * 25
        y = 400
        if x <= 780:  # Window width - car width
            draw_car(window, x, y, car.colour)

    # Top lane: cars from y=0 to y=225, x=350
    for i, car in enumerate(traffic.top):
        y = 225 - i * 25
        x = 350
        if y >= 0:
            draw_car(window, x, y, car.colour)

    # Bottom lane: cars from y=600 to y=425, x=450
    for i, car in enumerate(traffic.bottom):
        y = 425 + i * 25
        x = 450
        if y <= 590:  # Window height - car height
            draw_car(window, x, y, car.colour)

    # Display car counts
    font = pygame.font.SysFont(None, 24)
    window.blit(font.render(f'Left: {len(traffic.left)}', False, WHITE), (50, 275))
    window.blit(font.render(f'Top: {len(traffic.right)}', False, WHITE), (425, 50))
    window.blit(font.render(f'Right: {len(traffic.top)}', False, WHITE), (700, 375))
    window.blit(font.render(f'Bottom: {len(traffic.bottom)}', False, WHITE), (325, 550))