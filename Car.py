import random
import time

class Car:
    def __init__(self, loc, dest, timeWaiting):
        self.loc = loc
        self.dest = dest
        self.timeWaiting = timeWaiting
        self.colour = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    def colour(self):
        return self.colour
