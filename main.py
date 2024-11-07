import pygame
import time
import csv
import random

from Car import *

def initBoard():
    bkgColour = (255, 255, 255)
    screen = pygame.display.set_mode((800, 600))
    screen.fill(bkgColour)
    pygame.display.flip()
    running = True

def drawBoard():
    pass

hours = 24
days = 30
gameLength = hours * days
gameLengthMultiplier = 0.7 # speeds up or slows down the game
carsOnBoard = []
cycle = 0

with open('kaggle/traffic.csv') as file:
    reader = csv.reader(file)
    file = list(reader)

initBoard()

for line in file:
    print(line)
    print("cycle number: " + str(cycle))
    for i in range(int(line[2])):
        carsOnBoard.append(Car(random.randint(1,4), random.randint(1,4), cycle, None))

    for event in pygame.event.get():
        # handles quit event
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    print()
    cycle += 1
    time.sleep(1 * gameLengthMultiplier)
