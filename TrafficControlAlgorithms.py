from MLAlgorithms import Controller

class ChaosControl(Controller):
    def choose_action(self, traffic):
        lanes = [len(traffic.left), len(traffic.right), len(traffic.top), len(traffic.bottom)]
        return lanes.index(max(lanes))

    def update(self, traffic, cars_moved, missed_cars):
        pass

class BetterControl(Controller):
    def __init__(self):
        self.cycle = 0

    def choose_action(self, traffic):
        mod = self.cycle % 8
        if 0 <= mod <= 1:
            action = 0
        elif 2 <= mod <= 3:
            action = 1
        elif 4 <= mod <= 5:
            action = 2
        else:  # 6 <= mod <= 7
            action = 3
        self.cycle += 1
        return action

    def update(self, traffic, cars_moved, missed_cars):
        pass

class SimpleControl(Controller):
    def __init__(self):
        self.cycle = 0

    def choose_action(self, traffic):
        mod = self.cycle % 12
        if 0 <= mod <= 2:
            action = 0
        elif 3 <= mod <= 5:
            action = 1
        elif 6 <= mod <= 8:
            action = 2
        else:  # 9 <= mod <= 11
            action = 3
        self.cycle += 1
        return action

    def update(self, traffic, cars_moved, missed_cars):
        pass