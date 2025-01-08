import pygame
import time
import csv
import random

from Car import *
from WindowManager import *
from TrafficControlAlgorithms import *

# initialise pygame and set window dimensions
pygame.init()
window_width, window_height = 800, 600
window = pygame.display.set_mode((window_width, window_height))

# reads csv of dataset and creates a list of the elements on each line
try:
	with open('kaggle/traffic.csv') as file:
		reader = csv.reader(file)
		file = list(reader)
except:
	print("something went wrong while trying to read dataset")
	quit()


gameLength = 24 * 30
gameLengthMultiplier = 1 # speeds up or slows down the game
carsOnBoard = []
carMultiplier = 1 # multiplies number of cars on board
hour = 0
cyclesPerHour = 60
roadNumbers = [1,2,3,4]
probability = [0.3,0.2,0.3,0.2]

running = True

# this is the main game loop, it iterates though the lines of the csv and
for line in file:
	numCarsThisCycle = line[2]
	print(numCarsThisCycle)
	print("hour number: " + str(hour))

	for cycle in range(cyclesPerHour):
		print("minute number: " + str(cycle))

		for i in range(int(line[2]) * carMultiplier):
			carsOnBoard.append(Car(random.choices(roadNumbers, probability), random.choices(roadNumbers, probability), 0))

		print(len(carsOnBoard))

		# this is a basic time based cycling of the lights, light cycles every couple of minutes
		# mode = 0 = left light
		# mode = 1 = right light
		# mode = 2 = top light
		# mode = 3 = bottom light
		mode = betterControl(cycle)
		match mode:
			case 0:
				print("left light")
			case 1:
				print("right light")
			case 2:
				print("top light")
			case 3:
				print("bottom light")

		# draw the window
		drawWindow(window, mode, carsOnBoard)
		for car in carsOnBoard:
			car.timeWaiting += 1

		for event in pygame.event.get():
			# handles quit event
			if event.type == pygame.QUIT:
				# dump all the car objects to a text file
				with open("dump.txt", "w") as dump_file:
					for car in carsOnBoard:
						print('c')
						dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")

				running = False
				pygame.quit()
				quit()

		time.sleep(1 * gameLengthMultiplier)

	print()
	hour += 1


