import random
import time

class Car:
    def __init__(self, loc, dest, leavetime, arrivaltime):
        self.loc = loc
        self.dest = dest
        self.leavetime = leavetime
        self.arrivaltime = arrivaltime
        self.colour = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    def colour(self):
        return self.colour


