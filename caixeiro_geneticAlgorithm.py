from random import random, randint, choices, shuffle
from math import dist
import matplotlib.pyplot as plt

# Calculates the total distance of a state, adding the distance between each point
def calculateTspDistance(state):
    totalDistance = 0
    for i in range(len(state) - 1):
        totalDistance += dist(state[i], state[i + 1])

    # Closed loop, add closing distance (distance between last and first point)
    totalDistance += dist(state[0], state[-1])
    return totalDistance

# Generates a possible state by swapping two random points on the current state
def generatePossibleState(currentState):
    randomPoint1 = randint(0, len(currentState) - 1)
    randomPoint2 = randint(0, len(currentState) - 1)
    while randomPoint1 == randomPoint2:
        randomPoint1 = randint(0, len(currentState) - 1)
        randomPoint2 = randint(0, len(currentState) - 1)

    possibleState = currentState.copy()
    possibleState[randomPoint1], possibleState[randomPoint2] = possibleState[randomPoint2], possibleState[randomPoint1]
    return possibleState

# Generate numberOfPoints points on a maxCoordinate x maxCoordinate grid, calculate the total distance between them and add them to an array. 
# Shuffle them and repeat populationSize times.
def generateStartingPopulation(populationSize, numberOfPoints, maxCoordinate):
    points = []
    for _ in range(numberOfPoints):
        points.append((randint(0, maxCoordinate), randint(0, maxCoordinate)))
    
    population = []
    for _ in range(populationSize):
        distance = calculateTspDistance(points)
        population.append((distance, points))
        # Instead of shuffling all points, generate a state based on the current state by swapping two random points
        # points = generatePossibleState(points)
        shuffle(points)
    
    return population

# Given two parents, creates a crossover and generates a child based on a randomly selected crossover point
def reproduce(parent1, parent2):
    parentLength = len(parent1[1])
    crossoverPoint = randint(1, parentLength)
    parent1cut = parent1[1][0:crossoverPoint]
    parent2cut = [point for point in parent2[1] if point not in parent1cut]
    crossover = parent1cut + parent2cut
    child = (calculateTspDistance(crossover), crossover)
    return child

# The main algorithm loop
def geneticAlgorithm(startingPopulation, maxSteps, mutationRate, adaptationThreshold):
    currentStep = 1
    population = startingPopulation
    currentBestState = sorted(startingPopulation)[0][1]
    currentBestStateDistance = sorted(startingPopulation)[0][0]
    totalBestState = currentBestState
    totalBestStateDistance = currentBestStateDistance
    bestArray = []
    distanceArray = []
    while(currentStep <= maxSteps):
        newPopulation = []
        # Possible roulette choice implementation
        # totalFitness = sum(state[0] for state in population)
        # selectionProbabilities = [(state[0] / totalFitness) for state in population]
        for _ in range (len(population)):
            # Possible roulette choice implementation
            # parent1 = sorted(choices(population, weights=selectionProbabilities))[0]
            # parent2 = sorted(choices(population, weights=selectionProbabilities))[0]
            parent1 = sorted(choices(population, k = 4))[0]
            parent2 = sorted(choices(population, k = 4))[0]
            child = reproduce(parent1, parent2)
            if random() < mutationRate:
                randomPoint1, randomPoint2 = randint(0, len(child[1]) - 1), randint(0, len(child[1]) - 1)
                child[1][randomPoint1], child[1][randomPoint2] = child[1][randomPoint2], child[1][randomPoint1]

            newPopulation.append(child)

        population = newPopulation
        currentBestState = sorted(population)[0][1]
        currentBestStateDistance = sorted(population)[0][0]
        if currentBestStateDistance < adaptationThreshold:
            totalBestState = currentBestState
            print(f"Ended by adaptation. Adaptation Threshold: {adaptationThreshold}. Total Distance: {currentBestStateDistance}.")
            break

        if currentBestStateDistance < totalBestStateDistance:
            print(f"Found a better candidate. Previous Total Distance: {totalBestStateDistance}. New Best Total Distance: {currentBestStateDistance}.")
            totalBestStateDistance = currentBestStateDistance
            bestArray.append(totalBestStateDistance)
            totalBestState = currentBestState
            
        distanceArray.append(currentBestStateDistance)
        currentStep += 1

    if(currentStep == maxSteps + 1):
        print(f"Ended by stepcount: {currentStep - 1} steps.")
    return totalBestState, bestArray, distanceArray

def main():
    populationSize = 50
    numberOfPoints = 15
    maxCoordinate = 100
    mutationRate = 0.1
    maxSteps = 5000
    adaptationThreshold = 325
    print(f"Starting simulation. Generated {numberOfPoints} points, will run {maxSteps} steps or until it reaches {adaptationThreshold} distance or lower.")

    startingPopulation = generateStartingPopulation(populationSize, numberOfPoints, maxCoordinate)
    bestState, bestArray, distanceArray = geneticAlgorithm(startingPopulation, maxSteps, mutationRate, adaptationThreshold)
    bestState.append(bestState[0])
    xb, yb = zip(*bestState)
    plt.subplot(1, 3, 1)
    plt.plot(xb, yb)
    plt.scatter(xb, yb, color="black")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title("Best State")
    
    plt.subplot(1, 3, 2)
    plt.plot(bestArray)
    plt.xlabel('Number of better states found')
    plt.ylabel('Total distance')
    plt.title("Total distance by findings of better states")

    plt.subplot(1, 3, 3)
    plt.plot(distanceArray)
    plt.xlabel('Total steps')
    plt.ylabel('Total distance')
    plt.title("Distance by steps")

    plt.show()
if __name__ == "__main__":
    main()
