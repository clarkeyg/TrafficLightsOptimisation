import random
import time

class Car:
    def __init__(self, loc, dest, timeWaiting):
        self.loc = loc
        self.dest = dest
        self.timeWaiting = timeWaiting
        self.colour = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

class Traffic:
    def __init__(self):
        self.top = []
        self.bottom = []
        self.left = []
        self.right = []
        self.liveCars = []
        self.deadCars = []

    def addCar(self, car):
        self.liveCars.append(car)
        if car.loc == [0]:
            self.left.append(car)
        elif car.loc == [1]:
            self.top.append(car)
        elif car.loc == [2]:
            self.right.append(car)
        elif car.loc == [3]:
            self.bottom.append(car)
        else:
            print("Invalid location")
