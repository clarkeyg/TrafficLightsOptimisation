import time
import csv
import random
import pygame
from Traffic import Car, Traffic
from WindowManager import draw_window
from MLAlgorithms import NeuralTrafficControl, ActorCriticTrafficControl, FixedNNControl
from TrafficControlAlgorithms import ChaosControl, SimpleControl, BetterControl

def quit_and_log(traffic, total_missed_cars, hour, cycle, cycles_per_hour):
	waiting_sum = 0
	live_count = 0
	max_wait = 0  # Initialize maximum wait time tracker
	with open("postReport.txt", "w") as dump_file:
		dump_file.write("Live Cars:\n")
		for lane in [traffic.top, traffic.bottom, traffic.left, traffic.right]:
			for car in lane:
				dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")
				waiting_sum += car.timeWaiting
				live_count += 1
				if car.timeWaiting > max_wait:  # Update max_wait if current car's wait time is higher
					max_wait = car.timeWaiting
		dump_file.write("\nDead Cars:\n")
		for car in traffic.deadCars:
			dump_file.write(f"{car.loc}{car.dest}{car.timeWaiting}{car.colour}\n")
			waiting_sum += car.timeWaiting
			if car.timeWaiting > max_wait:  # Update max_wait for dead cars
				max_wait = car.timeWaiting
		avg_wait = waiting_sum / (len(traffic.deadCars) + live_count) if (len(traffic.deadCars) + live_count) > 0 else 0
		dump_file.write(f"Avg waiting time: {avg_wait} cycles\n")
		dump_file.write(f"Max waiting time: {max_wait} cycles\n")  # Log the maximum wait time
		dump_file.write(f"Total potential cars wasted: {total_missed_cars}\n")
		total_cycles = (hour * cycles_per_hour) + cycle + 1  # +1 since cycle is 0-based
		dump_file.write(f"Avg potential cars wasted: {total_missed_cars / total_cycles if total_cycles > 0 else 0}\n")
	pygame.quit()
	quit()

# Initialise pygame
pygame.init()
window_width, window_height = 800, 600
window = pygame.display.set_mode((window_width, window_height))

# Load traffic data
try:
	with open('kaggle/traffic.csv') as file:
		reader = csv.reader(file)
		traffic_data = list(reader)
except:
	print("Error reading dataset")
	quit()

# Configuration
algorithm = "simple control"
game_length = 48
game_length_multiplier = 0.1
traffic = Traffic()
car_multiplier = 3
hour = 0
cycles_per_hour = 60
lower_limit_cars = 6
upper_limit_cars = 10
total_missed_cars = 0
road_numbers = [0, 1, 2, 3]
probability = [0.2, 0.4, 0.3, 0.1]

# Select controller
if algorithm == "neural online":
	controller = NeuralTrafficControl()
elif algorithm == "actor critic":
	controller = ActorCriticTrafficControl()
elif algorithm == "neural fixed":
	controller = FixedNNControl()
elif algorithm == "chaos control":
	controller = ChaosControl()
elif algorithm == "simple control":
	controller = SimpleControl()
elif algorithm == "better control":
	controller = BetterControl()
else:
	print("Invalid algorithm")
	quit()

# Start logging
with open("liveLog.txt", "w") as log:
	log.write(f"Log Session started\n\nSettings:\n"
	          f"gameLength {game_length}\ncarMultiplier {car_multiplier}\n"
	          f"cyclesPerHour {cycles_per_hour}\nlowerLimitCars {lower_limit_cars}\n"
	          f"upperLimitCars {upper_limit_cars}\n\n")

	running = True
	for line in traffic_data:
		num_cars_this_hour = int(line[2]) * car_multiplier
		print(f"cars per hour: {num_cars_this_hour}")

		for cycle in range(cycles_per_hour):
			print(f"hour number: {hour}\ncycle number: {cycle}")
			log.write(f"hour number: {hour}\nminute number: {cycle}\n")

			num_cars_this_cycle = round(num_cars_this_hour / cycles_per_hour)
			for _ in range(num_cars_this_cycle):
				traffic.addCar(Car(random.choices(road_numbers, probability),
				                   random.choices(road_numbers, probability), 0))

			# Choose action using controller
			mode = controller.choose_action(traffic)

			# Move cars
			missed_cars, cars_moved = 0, 0

			if mode == 0:
				print("left light")
				log.write("left light\n")
				for _ in range(random.randint(lower_limit_cars, upper_limit_cars)):
					try:
						traffic.deadCars.append(traffic.left.pop())
						cars_moved += 1
					except IndexError:
						missed_cars += 1
						total_missed_cars += 1
			elif mode == 1:
				print("right light")
				log.write("right light\n")
				for _ in range(random.randint(lower_limit_cars, upper_limit_cars)):
					try:
						traffic.deadCars.append(traffic.right.pop())
						cars_moved += 1
					except IndexError:
						missed_cars += 1
						total_missed_cars += 1
			elif mode == 2:
				print("top light")
				log.write("top light\n")
				for _ in range(random.randint(lower_limit_cars, upper_limit_cars)):
					try:
						traffic.deadCars.append(traffic.top.pop())
						cars_moved += 1
					except IndexError:
						missed_cars += 1
						total_missed_cars += 1
			elif mode == 3:
				print("bottom light")
				log.write("bottom light\n")
				for _ in range(random.randint(lower_limit_cars, upper_limit_cars)):
					try:
						traffic.deadCars.append(traffic.bottom.pop())
						cars_moved += 1
					except IndexError:
						missed_cars += 1
						total_missed_cars += 1

			print(f"moved cars: {cars_moved}\nmissed cars: {missed_cars}")
			log.write(f"moved cars: {cars_moved}\nmissed cars: {missed_cars}\n")

			# Update waiting times
			for car in traffic.top + traffic.bottom + traffic.left + traffic.right:
				car.timeWaiting += 1

			# Render
			draw_window(window, mode, traffic)
			log.write("\n")

			# Update controller
			controller.update(traffic, cars_moved, missed_cars)

			# Check end condition
			if hour == game_length and cycle == cycles_per_hour - 1:
				quit_and_log(traffic, total_missed_cars, hour, cycle, cycles_per_hour)

			# Handle quit event
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					quit_and_log(traffic, total_missed_cars, hour, cycle, cycles_per_hour)

			print()
			time.sleep(1 * game_length_multiplier)

		log.write("\n")
		hour += 1