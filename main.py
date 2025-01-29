"""
using Python 3.12
"""

import time
import csv
import random

from Traffic import Car, Traffic
from WindowManager import *
from TrafficControlAlgorithms import *
from MLAlgorithms import *

def quitAndLog():
	waitingSum = 0
	liveCount = 0
	# dump all the car objects to a text file
	with open("dump.txt", "w") as dump_file:

		dump_file.write("Live Cars:\n")
		for car in traffic.top:
			dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")
			waitingSum += car.timeWaiting
			liveCount += 1
		for car in traffic.bottom:
			dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")
			waitingSum += car.timeWaiting
			liveCount += 1
		for car in traffic.left:
			dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")
			waitingSum += car.timeWaiting
			liveCount += 1
		for car in traffic.right:
			dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")
			waitingSum += car.timeWaiting
			liveCount += 1

		dump_file.write("\nDead Cars:\n")
		for car in traffic.deadCars:
			dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")
			waitingSum += car.timeWaiting

		dump_file.write(f"Avg waiting time: {waitingSum / (len(traffic.deadCars) + liveCount)}\n")
		dump_file.write(f"Total potential cars wasted: {totalMissedCars}\n")
		dump_file.write(f"Avg potential cars wasted: {totalMissedCars / ((hour * cyclesPerHour) + cycle)}\n")

	running = False
	pygame.quit()
	quit()

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


gameLength = 24 # in hours
gameLengthMultiplier = 0.01 # speeds up or slows down the game
traffic = Traffic()
carMultiplier = 1 # multiplies number of cars on board
hour = 0
cyclesPerHour = 30
lowerLimitCars = 15
upperLimitCars = 20
totalMissedCars = 0

# 0 = left light
# 1 = right light
# 2 = top light
# 3 = bottom light
# respective probabilities of cars coming from each road (simulates some roads being busier than others)
roadNumbers = [	0,	1,	2,	3]
probability = [0.2,0.4,0.3,0.1]

running = True

# this is the main game loop, it iterates though the lines of the csv
for line in file:
	numCarsThisCycle = line[2]
	print(f"cars per hour{numCarsThisCycle}")

	for cycle in range(cyclesPerHour):
		print(f"hour number: {str(hour)}")
		print("minute number: " + str(cycle))

		for i in range(int(numCarsThisCycle) * carMultiplier):
			traffic.addCar(Car(random.choices(roadNumbers, probability), random.choices(roadNumbers, probability), 0))

		# this is the algorithm that decides which light to turn green
		# it does this by setting the traffic light "mode"
		mode =  simpleControl(cycle)

		# pop cars from each light during cycle
		# mode = 0 = left light
		# mode = 1 = right light
		# mode = 2 = top light
		# mode = 3 = bottom light
		missedCars = 0
		match mode:
			case 0:
				print("left light")
				for i in range(random.randint(lowerLimitCars, upperLimitCars)):
					try:
						traffic.deadCars.append(traffic.left.pop())
					except:
						missedCars += 1
						totalMissedCars += 1
			case 1:
				print("right light")
				for i in range(random.randint(lowerLimitCars, upperLimitCars)):
					try:
						traffic.deadCars.append(traffic.right.pop())
					except:
						missedCars += 1
						totalMissedCars += 1
			case 2:
				print("top light")
				for i in range(random.randint(lowerLimitCars, upperLimitCars)):
					try:
						traffic.deadCars.append(traffic.top.pop())
					except:
						missedCars += 1
						totalMissedCars += 1
			case 3:
				print("bottom light")
				for i in range(random.randint(lowerLimitCars, upperLimitCars)):
					try:
						traffic.deadCars.append(traffic.bottom.pop())
					except:
						missedCars += 1
						totalMissedCars += 1


		print(f"missed cars: {missedCars}")

		# increment timeWaiting for all cars in traffic
		for car in traffic.top:
			car.timeWaiting += 1
		for car in traffic.bottom:
			car.timeWaiting += 1
		for car in traffic.left:
			car.timeWaiting += 1
		for car in traffic.right:
			car.timeWaiting += 1

		# draw the window
		drawWindow(window, mode, traffic)

		if (hour == gameLength and cycle == 29):
			quitAndLog()

		for event in pygame.event.get():
			# handles quit event
			if event.type == pygame.QUIT:
				quitAndLog()

		print()
		time.sleep(1 * gameLengthMultiplier)

	hour += 1


