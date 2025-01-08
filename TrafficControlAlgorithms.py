
# mode = 0 = left light
# mode = 1 = right light
# mode = 2 = top light
# mode = 3 = bottom light

def simpleControl(cycle):
    return cycle % 4

def betterControl(cycle):
    thing = cycle % 8
    if thing >=0 and thing <= 1:
        return 0
    elif thing >= 2 and thing <= 3:
        return 1
    elif thing >= 4 and thing <= 5:
        return 2
    elif thing >= 6 and thing <= 7:
        return 3