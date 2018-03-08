# -----------------------------------------------------------------------------
#
# Car/A.I. Learns to Drive
#
# Language - Python
# Modules - pygame, sys, random, numpy, pickle, cPickle, math
#
# By - Jatin Kumar Mandav
#
# Website - https://jatinmandav.wordpress.com
#
# YouTube Channel - https://www.youtube.com/mandav
# GitHub - github.com/jatinmandav
# Twitter - @jatinmandav
#
# -----------------------------------------------------------------------------

import pygame
import sys
from math import *

import random
import numpy as np

import pickle
import cPickle

pygame.init()

width = 1000
height = 620
display = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

font = pygame.font.SysFont("Agency FB", 20)
font2 = pygame.font.SysFont("Agency FB", 30)

mutationRate = 5
generation = 0

bestSoFar = 0
bestGeneration = 0


# Maps or Tracks
class Map:
    def __init__(self):
        self.path = []
        self.mapWidth = 10
        self.mapHeight = 10
        self.start = (0, 0)
        self.blockSize = 60
        
    def createMap(self):
        loop = True

        while loop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    close()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        close()
                    if event.key == pygame.K_s:
                        loop = False
                        print(self.path)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    i = pos[0]/self.blockSize
                    j = pos[1]/self.blockSize
                    if (i, j) in self.path:
                        self.path.remove((i, j))
                    else:
                        self.path.append((i, j))

                    
            display.fill((28, 40, 51))

            text = font2.render("Create Map", True, (208, 211, 212))
            display.blit(text, (width - 300, 100))

            text = font2.render("Press S to Start", True, (208, 211, 212))
            display.blit(text, (width - 300, 150))

            self.drawMap()
        
            pygame.display.update()
            clock.tick()
            
        
        
    def drawMap(self):
        pygame.draw.rect(display, (236, 240, 241), (self.start[0], self.start[1], self.blockSize*self.mapWidth, self.blockSize*self.mapHeight))
        for i in xrange(self.mapHeight):
            for j in xrange(self.mapWidth):
                if (i, j) in self.path:
                    pygame.draw.rect(display, (236, 240, 241), (self.start[0] + (i+1)*1 + i*(self.blockSize - 1), self.start[1] + (j+1)*1 + j*(self.blockSize - 1),
                                                         self.blockSize - 2, self.blockSize - 2))
                else:
                    pygame.draw.rect(display, (52, 73, 94), (self.start[0] + (i+1)*1 + i*(self.blockSize - 1), self.start[1] + (j+1)*1 + j*(self.blockSize - 1),
                                                         self.blockSize - 2, self.blockSize - 2))

        if not self.path == []:
            pygame.draw.rect(display, (17, 122, 101), (self.start[0] + (self.path[0][0]+1)*1 + self.path[0][0]*(self.blockSize - 1),
                                                      self.start[1] + (self.path[0][1]+1)*1 + self.path[0][1]*(self.blockSize - 1),
                                                             self.blockSize - 2, self.blockSize - 2))

            pygame.draw.rect(display, (203, 67, 53), (self.start[0] + (self.path[-1][0]+1)*1 + self.path[-1][0]*(self.blockSize - 1),
                                                      self.start[1] + (self.path[-1][1]+1)*1 + self.path[-1][1]*(self.blockSize - 1),
                                                             self.blockSize - 2, self.blockSize - 2))

# Brain or the Neural Network Implementation using Numpy
class Brain:
    def __init__(self, size):
        self.size = size
        self.weights = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
        self.biases = [np.random.randn(y, 1) for y in size[1:]]

    def feedForward(self, data):
        i = 0
        for b, w in zip(self.biases, self.weights):
            activation = np.dot(w, data) + b
            if i == 0:
                data = ReLU(activation)
                data = softmax(data)
            else:
                data = sigmoid(activation)
            i += 1
        data = softmax(data)

        controller = []

        for i in xrange(len(data)): 
            controller.append(np.max(data[i]))

        val = np.argmax(controller)

        if val == 0:
            return 1
        elif val == 1:
            return 0
        else:
            return -1

# ReLU activation Function
def ReLU(data):
    data[data < 0] = 0

    return data

# Sigmoid Activation Function
def sigmoid(data):
    return 1.0/(1.0 + np.exp(-data))

# Softmax or averaging the value
def softmax(data):
    summation = np.sum(data)

    if summation == 0.0:
        summation = 1.0
    
    for i in xrange(len(data)):
        data[i] = data[i]/summation

    return data

# Cars
class Car:
    def __init__(self, track, color=(241, 196, 15)):
        self.x = (track.path[0][0])*track.blockSize + track.blockSize
        self.y = (track.path[0][1])*track.blockSize + track.blockSize/2
        self.w = 40
        self.h = 25
        self.angle = 0
        self.changeAngle = 5
        self.coord = []
        self.speed = 2

        self.color = color

        self.sensor1 = [0, 0]
        self.sensor2 = [0, 0]
        self.sensor3 = [0, 0]
        self.sensor4 = [0, 0]

        self.brain = Brain([4, 5, 3])
        self.crashed = False

        self.completed = False

        self.score = 0
        self.fitness = 0
        self.prob = 0

    def reset(self):
        self.x = 30
        self.y = 30
        self.w = 40
        self.h = 25
        self.angle = 0

        self.score = 0.0
        self.crashed = False
        
    def translate(self, coord):
        return [coord[0] + self.x, coord[1] + self.y]

    def rotate(self, coord, angle, anchor=(0, 0)):
        corr = 180
        return ((coord[0] - anchor[0])*cos(angle + radians(corr)) - (coord[1] - anchor[1])*sin(angle + radians(corr)),
                (coord[0] - anchor[0])*sin(angle + radians(corr)) + (coord[1] - anchor[1])*cos(angle + radians(corr)))

    def think(self):
        inputBrain = []
        # Calculting the distance in sensors
        inputBrain.append(((self.x - self.sensor1[0])**2 + (self.y - self.sensor1[1])**2)**0.5)
        inputBrain.append(((self.x - self.sensor2[0])**2 + (self.y - self.sensor2[1])**2)**0.5)
        inputBrain.append(((self.x - self.sensor3[0])**2 + (self.y - self.sensor3[1])**2)**0.5)
        inputBrain.append(((self.x - self.sensor4[0])**2 + (self.y - self.sensor4[1])**2)**0.5)

        result = self.brain.feedForward(inputBrain)
    
        self.angle += result*self.changeAngle
        
    def move(self, track):
        if not self.crashed:
            self.x = self.x + self.speed*cos(radians(self.angle))
            self.y = self.y + self.speed*sin(radians(self.angle))

            self.score += 0.06

            self.sensor1 = [self.x, self.y]
            self.sensor2 = [self.x, self.y]
            self.sensor3 = [self.x, self.y]
            self.sensor4 = [self.x, self.y]
            
            # Finding the end point detected by sensors
            while True:
                self.sensor1[0] += 1*cos(radians(self.angle - 90))
                self.sensor1[1] += 1*sin(radians(self.angle - 90))
                if not (((int(self.sensor1[0])/track.blockSize, int(self.sensor1[1])/track.blockSize)) in track.path):
                    break
            while True:
                self.sensor2[0] += 1*cos(radians(self.angle - 45))
                self.sensor2[1] += 1*sin(radians(self.angle - 45))
                if not (((int(self.sensor2[0])/track.blockSize, int(self.sensor2[1])/track.blockSize)) in track.path):
                    break
            while True:
                self.sensor3[0] += 1*cos(radians(self.angle + 45))
                self.sensor3[1] += 1*sin(radians(self.angle + 45))
                if not (((int(self.sensor3[0])/track.blockSize, int(self.sensor3[1])/track.blockSize)) in track.path):
                    break

            while True:
                self.sensor4[0] += 1*cos(radians(self.angle + 90))
                self.sensor4[1] += 1*sin(radians(self.angle + 90))
                if not (((int(self.sensor4[0])/track.blockSize, int(self.sensor4[1])/track.blockSize)) in track.path):
                    break

            points = [(0, 0), (0, self.h), (self.w, self.h), (self.w, 0)]
            self.coord = []
            for point in points:
                self.coord.append(self.translate(self.rotate(point, radians(self.angle), (self.w/2, self.h/2))))

            for point in self.coord:
                if (int(point[0])/track.blockSize, int(point[1])/track.blockSize) == track.path[-1]:
                    self.completed = True
                if not ((int(point[0])/track.blockSize, int(point[1])/track.blockSize) in track.path):
                    self.crashed = True
                    self.fitness = self.score
                    
            self.think()
            
    def draw(self):
        sensorColor = (46, 64, 83)
        pygame.draw.line(display, sensorColor, (self.x, self.y), self.sensor1)
        pygame.draw.line(display, sensorColor, (self.x, self.y), self.sensor2)
        pygame.draw.line(display, sensorColor, (self.x, self.y), self.sensor3)
        pygame.draw.line(display, sensorColor, (self.x, self.y), self.sensor4)

        pygame.draw.ellipse(display, sensorColor, (self.sensor1[0], self.sensor1[1], 5, 5))
        pygame.draw.ellipse(display, sensorColor, (self.sensor2[0], self.sensor2[1], 5, 5))
        pygame.draw.ellipse(display, sensorColor, (self.sensor3[0], self.sensor3[1], 5, 5))
        pygame.draw.ellipse(display, sensorColor, (self.sensor4[0], self.sensor4[1], 5, 5))
        
        pygame.draw.polygon(display, self.color, self.coord)

    def displayInfo(self, pos):
        text = font.render("Score: " + str("{0:.2f}".format(self.score)), True, self.color)
        display.blit(text, pos)

# Population Pool
class Population:
    def __init__(self):
        self.population = []
        self.crashed = []

    def createPopulation(self, track):
        color = [(231, 76, 60), (142, 68, 173), (52, 152, 219),
                 (22, 160, 133), (241, 196, 15), (211, 84, 0),
                 (81, 90, 90), (100, 30, 22), (125, 102, 8),
                 (26, 82, 118)]

        for i in xrange(len(color)):
            car = Car(track, color[i])
            self.population.append(car)
    def loadData(self):
        # Loading the Previous generation Weights and Biases
        index = 0
        for i in xrange(len(self.population)):
            with open("Trained_biases/biases_" + str(index) + ".pkl", "rb") as f:
                self.population[i].brain.biases = pickle.load(f)
            with open("Trained_weights/weights_" + str(index) + ".pkl", "rb") as f:
                self.population[i].brain.weights = pickle.load(f)
            index += 1

    def loadBest(self):
        # Loading the Best Weights and Biases So Far
        index = 0
        for i in xrange(len(self.population)):
            with open("Trained_biases_Best/biases_" + str(index) + ".pkl", "rb") as f:
                self.population[i].brain.biases = pickle.load(f)
            with open("Trained_weights_Best/weights_" + str(index) + ".pkl", "rb") as f:
                self.population[i].brain.weights = pickle.load(f)
            index += 1

    def move(self, track):
        for car in self.population:
            car.move(track)

    def evolve(self, track):
        for car in self.population:
            if not car.crashed:
                return
        self.reproduction(track)

    def reproduction(self, track):
        global generation, bestGeneration, bestSoFar
        generation += 1
        
        self.normalizeFitness()

        fitness = 0
        for car in self.population:
            if car.fitness > fitness:
                fitness = car.fitness

        bestGeneration = fitness

        if bestGeneration > bestSoFar:
            bestSoFar = bestGeneration
            save_best_generation(self.population)

        newGen = Population()

        i = 0

        while len(newGen.population) < len(self.population):
            indexA = pickOne(self.population[:])
            indexB = pickOne(self.population[:])

            childA = Car(track, self.population[i].color)
            i += 1
            childB = Car(track, self.population[i].color)
            i += 1
            
            childA.brain, childB.brain = self.crossover(self.population[indexA].brain, self.population[indexB].brain)
        
            childA.brain = self.mutation(childA.brain)
            childB.brain = self.mutation(childB.brain)

            newGen.population.append(childA)
            newGen.population.append(childB)

        self.population = newGen.population[:]

    def crossover(self, brainA, brainB):
        newBrain1 = Brain(brainA.size)
        newBrain2 = Brain(brainA.size)
        
        for i in xrange(len(brainA.weights)):
            for j in xrange(len(brainA.weights[i])):
                for k in xrange(len(brainA.weights[i][j])):
                    if random.randint(0, 1):                        
                        newBrain1.weights[i][j][k] = brainA.weights[i][j][k]
                        newBrain2.weights[i][j][k] = brainB.weights[i][j][k]
                    else:
                        newBrain1.weights[i][j][k] = brainB.weights[i][j][k]
                        newBrain2.weights[i][j][k] = brainA.weights[i][j][k]

        return newBrain1, newBrain2

    def mutation(self, brain):
        prob = random.randint(1, 100)

        if prob <= mutationRate:
            for i in xrange(len(brain.weights)):
                for j in xrange(len(brain.weights[i])):
                    k = random.randint(0, len(brain.weights[i][j]) - 1)
                    val = np.random.randn(1)[0]
                    brain.weights[i][j][k] = val

        return brain
    
    def normalizeFitness(self):
        summation = 0

        for car in self.population:
            summation += car.fitness

        for i in xrange(len(self.population)):
            if self.population[i].completed:
                self.population[i].prob = 0.9
            else:
                self.population[i].prob = self.population[i].fitness/summation
        
    def draw(self):
        for car in self.population:
            car.draw()

    def displayInfo(self):
        text = font2.render("Generation: " + str(generation), True, (240, 243, 244))
        display.blit(text, (width - 375, 100))
        text = font2.render("Best So Far: " + str(bestSoFar), True, (240, 243, 244))
        display.blit(text, (width - 375, 150))
        text = font2.render("Best Last Generation: " + str(bestGeneration), True, (240, 243, 244))
        display.blit(text, (width - 375, 200))

        text = font2.render("R -> Reset", True, (240, 243, 244))
        display.blit(text, (width - 350, height - 250))
        text = font2.render("S -> Save Current Generation", True, (240, 243, 244))
        display.blit(text, (width - 350, height - 200))
        text = font2.render("L -> Load Previous Best Generation", True, (240, 243, 244))
        display.blit(text, (width - 350, height - 150))
        text = font2.render("B -> Load The Best Generation Ever", True, (240, 243, 244))
        display.blit(text, (width - 350, height - 100))
        

        i = 0
        for car in self.population:
            car.displayInfo((width - 100, 50 + i*25))
            i += 1

# Picking Parents based on their fitness score
def pickOne(population):
    prob = random.uniform(0, 1)

    for i in xrange(len(population)):
        prob -= population[i].prob
        if prob < 0:
            return i

# Saving the Weights and Biases
def save_best_generation(population):
    i = 0
    for species in population:
        with open("Trained_biases/biases_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(species.brain.biases, f)
        with open("Trained_weights/weights_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(species.brain.weights, f)
        i += 1
    index = 0
    for i in xrange(len(population)):
        if population[i].score > population[index].score:
            index = i

    save_best_brain(population[index].brain.biases, population[index].brain.weights, population[index].score)
    

def save_best_brain(biases, weights, score):
    fp = open("bestScore", "r")
    prevScore = float(fp.read())
    fp.close()

    if score > prevScore:
        fp = open("bestScore", "w")
        fp.write(str(score))
        fp.close()
        with open("best_biases.pkl", "wb") as f:
            pickle.dump(biases, f)
        with open("best_weights.pkl", "wb") as f:
            pickle.dump(weights, f)


def close():
    pygame.quit()
    sys.exit()

def mainLoop():
    global generation, bestSoFar, bestGeneration
    bestSoFar = 0
    bestGeneration = 0
    generation = 0
    loop = True

    track = Map()
    track.createMap()

    population = Population()
    population.createPopulation(track)

    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    close()
                if event.key == pygame.K_r:
                    mainLoop()
                if event.key == pygame.K_s:
                    save_best_generation(population.population)
                if event.key == pygame.K_l:
                    population.loadData()
                if event.key == pygame.K_b:
                    population.loadBest()
            
        display.fill((28, 40, 51))

        track.drawMap()

        population.move(track)
        population.draw()
        population.evolve(track)
        population.displayInfo()
        
        pygame.display.update()
        clock.tick(60)

mainLoop()
