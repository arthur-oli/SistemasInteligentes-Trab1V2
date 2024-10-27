from random import randint, random
from math import exp, dist
import matplotlib.pyplot as plt

# Generates "numberOfPoints" random points in a maxCoordinate X maxCoordinate grid
def generateStartingPoints (numberOfPoints, maxCoordinate):
    points = []

    for _ in range (numberOfPoints):
        points.append((randint(0, maxCoordinate), randint(0, maxCoordinate)))

    return points

# Generates a random possible neighbor state by randomly swapping two points in the current state
def generatePossibleState(currentState):
    randomPoint1 = randint(0, len(currentState) - 1)
    randomPoint2 = randint(0, len(currentState) - 1)
    while (randomPoint1 == randomPoint2):
        randomPoint1 = randint(0, len(currentState) - 1)
        randomPoint2 = randint(0, len(currentState) - 1)
    possibleState = currentState.copy()
    possibleState[randomPoint1], possibleState[randomPoint2] = possibleState[randomPoint2], possibleState[randomPoint1]
    return possibleState

# Calculates the total distance of a state, adding the distance between each point
def calculateTspDistance(state):
    totalDistance = 0
    for i in range (len(state) - 1):
        totalDistance += dist(state[i], state[i + 1])
    
    # Closed loop, add closing distance (distance between last and first point)
    totalDistance += dist(state[0], state[-1])
    return totalDistance

# The algorithm itself. Given a starting state, runs until conditions are met and returns the best state found based on minimal total distance
def simulatedAnnealing(startingState, maxTemperature, minTemperature, startingTotalDistance, numberOfPoints):
    currentStep = 1
    currentState = startingState
    currentTotalDistance = startingTotalDistance
    currentTemperature = maxTemperature
    bestTotalDistance = startingTotalDistance
    bestState = startingState
    bestArray = []
    distanceArray = []
    maxSteps = 10 ** ((numberOfPoints / 10) + 2)
    # While the "temperature" (currentTemperature) is not low enough and the algorithm didn't reach maximum "time" (maxSteps)
    while(currentTemperature >= minTemperature):
        possibleState = generatePossibleState(currentState)
        possibleStateTotalDistance = calculateTspDistance(possibleState)
        deltaDistance = possibleStateTotalDistance - currentTotalDistance

        # Check if the neighbor state is better
        if possibleStateTotalDistance < bestTotalDistance:
            bestTotalDistance = possibleStateTotalDistance
            currentTotalDistance = bestTotalDistance
            bestArray.append(bestTotalDistance)
            bestState = possibleState.copy()
            currentState = bestState.copy()

        # If it isn't, check for random probability of making a locally bad choice
        elif random() < exp(-deltaDistance / currentTemperature):
            currentTotalDistance = possibleStateTotalDistance
            currentState = possibleState.copy()

        # Update the currentTemperature with a additive linear function (can be changed to exponential, logarithmic, etc)
        currentTemperature = minTemperature + (maxTemperature - minTemperature) * ((maxSteps - currentStep) / maxSteps)

        distanceArray.append(currentTotalDistance)
        currentStep += 1
    
    return bestState, bestArray, distanceArray

def main(): 
    # Simulated annealing parameters. Changing the numberOfPoints, temperature and number of steps (maxSteps) changes the outcome
    numberOfPoints = 20
    pointMaxCoordinate = 100
    maxTemperature = 10.0
    minTemperature = 0.001
    # Maximum steps depends on number of points. Exponential function picked based on testing.
    # Generate a starting state with random points
    startingState = generateStartingPoints(numberOfPoints, pointMaxCoordinate)
    startingTotalDistance = calculateTspDistance(startingState)
    
    # Run the main algorithm and save the best state found
    bestState, bestArray, distanceArray = simulatedAnnealing(startingState, maxTemperature, minTemperature, startingTotalDistance, numberOfPoints)

    # Plot the starting points (random points)
    startingState.append(startingState[0])
    xs, ys = zip(*startingState)
    plt.subplot(1, 5, 1)
    plt.scatter(xs, ys, color = 'black')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title("Generated Points")

    # Plot the starting state (random path)
    plt.subplot(1, 5, 2)
    plt.plot(xs, ys)
    plt.scatter(xs, ys, color = 'black')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title("Random Starting State")
    
    # Plot the best state found by the algorithm
    bestState.append(bestState[0])
    xb, yb = zip(*bestState)
    plt.subplot(1, 5, 3)
    plt.plot(xb, yb)
    plt.scatter(xb, yb, color = 'black')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title("Best State")

    plt.subplot(1, 5, 4)
    plt.plot(bestArray)
    plt.xlabel('Number of better states found')
    plt.ylabel('Total distance')
    plt.title("Total distance by findings of better states")

    plt.subplot(1, 5, 5)
    plt.plot(distanceArray)
    plt.xlabel('Total steps')
    plt.ylabel('Total distance')
    plt.title("Distance by steps")

    plt.show()

if __name__ == "__main__":
    main()