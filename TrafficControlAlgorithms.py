# mode = 0 = left light
# mode = 1 = right light
# mode = 2 = top light
# mode = 3 = bottom light

def simpleControl(cycle):
    return cycle % 4

def betterControl(cycle):
    thing = cycle % 8
    if 0 <= thing <= 1:
        return 0
    elif 2 <= thing <= 3:
        return 1
    elif 4 <= thing <= 5:
        return 2
    elif 6 <= thing <= 7:
        return 3

def chaosControl(traffic): #returns the lane that has most cars
    # get the number of cars at each light
    leftCars = len(traffic.left)
    rightCars = len(traffic.right)
    topCars = len(traffic.top)
    bottomCars = len(traffic.bottom)

    lanes = [leftCars, rightCars, topCars, bottomCars]

    maxTraffic = -1
    i = -1
    busiestLane = -1

    for currentTraffic in lanes:
        i += 1
        if currentTraffic > maxTraffic:
            maxTraffic = currentTraffic
            busiestLane = i

    print(busiestLane)

    return busiestLane

