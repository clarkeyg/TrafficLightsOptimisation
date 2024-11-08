import pygame
import time
import csv
import random

from Car import *

# initialise pygame and set window dimensions
pygame.init()
window_width, window_height = 800, 600
window = pygame.display.set_mode((window_width, window_height))

# pre-define some colours
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def draw_traffic_lights(x, y, mode):
	light_color = GREEN if mode == 0 else RED

	# traffic light background
	pygame.draw.rect(window, BLACK, (x, y, 20, 60))

	# draws light green or red depending on mode
	pygame.draw.circle(window, light_color, (x + 10, y + 15), 10)

def draw_intersection(mode):
	window.fill(WHITE)

	# draws the roads
	pygame.draw.rect(window, GRAY, (300, 0, 200, 600))  # Vertical road
	pygame.draw.rect(window, GRAY, (0, 225, 800, 200))  # Horizontal road

	# draws the traffic lights
	draw_traffic_lights(280, 250, mode - 0)  # Left
	draw_traffic_lights(500, 350, mode - 1)  # Right
	draw_traffic_lights(450, 200, mode - 2)  # Top
	draw_traffic_lights(350, 400, mode - 3)  # Bottom

def draw_cars(carsOnBoard):
	for car in carsOnBoard:
		if car.loc == 1:
			font = pygame.font.SysFont(None, 24)
			text = font.render(f'Cars: {len([c for c in carsOnBoard if c.loc == 1])}', False, WHITE)
			window.blit(text, (175, 275))
		elif car.loc == 2:
			font = pygame.font.SysFont(None, 24)
			text = font.render(f'Cars: {len([c for c in carsOnBoard if c.loc == 2])}', False, WHITE)
			window.blit(text, (425, 150))
		elif car.loc == 3:
			font = pygame.font.SysFont(None, 24)
			text = font.render(f'Cars: {len([c for c in carsOnBoard if c.loc == 3])}', False, WHITE)
			window.blit(text, (550, 375))
		elif car.loc == 4:
			font = pygame.font.SysFont(None, 24)
			text = font.render(f'Cars: {len([c for c in carsOnBoard if c.loc == 4])}', False, WHITE)
			window.blit(text, (325, 475))


gameLength = 24 * 30
gameLengthMultiplier = 0.7 # speeds up or slows down the game
carsOnBoard = []
carMultiplier = 1 # multiplies number of cars on board
hour = 0

# reads csv of dataset and creates a list of the elements on each line
try:
	with open('kaggle/traffic.csv') as file:
		reader = csv.reader(file)
		file = list(reader)
except:
	print("something went wrong while trying to read dataset")
	quit()

running = True

# this is the main game loop, it iterates though the lines of the csv and
for line in file:
	print(line)
	print("hour number: " + str(hour))

	for min in range(60):
		print("minute number: " + str(min))

		for i in range(int(line[2]) * carMultiplier):
			carsOnBoard.append(Car(random.randint(1, 4), random.randint(1, 4), 0))

		print(len(carsOnBoard))

		# this is a basic time based cycling of the lights, light cycles every minute
		mode = min % 4

		# draws the intersection and traffic lights based on the mode
		draw_intersection(mode)
		draw_cars(carsOnBoard)
		pygame.display.update()

		for event in pygame.event.get():
			# handles quit event
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		time.sleep(1 * gameLengthMultiplier)

	print()
	hour += 1


