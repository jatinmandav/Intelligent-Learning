# -----------------------------------------------------------------------------
#
# A.I. Learns to Play Flappy-Bird
#
# Language - Python
# Modules - pygame, sys, random, numpy, pickle, cPickle
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
import random

import numpy as np

import pickle
import cPickle

pygame.init()

width = 800
height = 400
display = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

pillar = None
generation = 0
bestScoreGeneration = 0
bestSoFar = 0
font = pygame.font.SysFont("Agency FB", 12)
font2 = pygame.font.SysFont("Agency FB", 20)

mutationRate = 30
speed  = 0
pillarGap = 300
index = 0

# Pillar or Obstacle
class Pillar:
    def __init__(self, start):
        self.gap = 150
        self.start = start
        self.x = start
        self.upperY = 0
        self.upperH = random.randint(20, height - 20 - self.gap)

        self.w = 30

        self.lowerY = random.randint(self.upperH + self.gap, height)
        self.lowerH = height - self.lowerY

        self.color = (144, 148, 151)

    def draw(self):
        pygame.draw.rect(display, self.color, (self.x, self.upperY, self.w, self.upperH))
        pygame.draw.rect(display, self.color, (self.x, self.lowerY, self.w, self.lowerH))

    def move(self):
        self.x -= 4

        if self.x < 0:
            self.reset()

    # Resetting the pillar
    def reset(self):
        index = 0
        for i in xrange(len(pillarSet)):
            if pillarSet[i].x > pillarSet[index].x:
                index = i
        
        self.x = pillarSet[index].x + pillarGap
        self.upperH = random.randint(20, height - 20 - self.gap)
        self.lowerY = random.randint(self.upperH + self.gap, height)
        self.lowerH = height - self.lowerY

def resetPillars():
    global index
    index = 0
    for i in xrange(len(pillarSet)):
        pillarSet[i].x = width + i*pillarGap

# Brain or the Neural Network Implementation using Numpy
class Brain:
    def __init__(self, size):
        self.size = size
        self.weights = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
        self.biases = [np.random.randn(y, 1) for y in size[1:]]

    def feedForward(self, data):
        i = 0
        for w, b in zip(self.weights, self.biases):
            activation = np.dot(w, data) + b
            if i == 0:
                data = ReLU(activation)
                data = softmax(data)
            else:
                data = sigmoid(activation)
            i += 1
        print(data)
    
        data = abs(data[0][0])
        
        if data > 0.5:
            return 1
        else:
            return 0

# ReLU activation Function
def ReLU(z):
    z[z < 0] = 0
    return z

# Softmax or averaging the value
def softmax(z):
    summation = np.sum(z)
    if summation == 0.0:
        summation = 1.0
    for i in xrange(len(z)):
        z[i] = z[i]/summation
    return z

# Sigmoid Activation Function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# Flappy Bird or Species
class Bird:
    def __init__(self, color):
        self.x = 200
        self.y = height/2 - 20
        self.w = 40
        self.h = 40
        self.color = color

        self.fitness = 0
        self.score = 0
        self.prob = 0

        self.life = True

        self.brain = Brain([5, 4, 1])

    def draw(self, pillar):
        pygame.draw.rect(display, self.color, (self.x, self.y, self.w, self.h))
        pygame.draw.line(display, self.color, (self.x + self.w/2, self.y + self.h/2), (pillar.x, pillar.upperH))
        pygame.draw.line(display, self.color, (self.x + self.w/2, self.y + self.h/2), (pillar.x, pillar.lowerY))

    def move(self, pillar):
        self.y += 8
        self.score += (1.0/60.0)
        self.think(pillar.x, pillar.upperH, pillar.lowerY)
        
        if not (-self.h < self.y < height):
            self.fitness = self.score
            self.life = False
            self.score = 0.0
        if self.y < pillar.upperH or self.y + self.h > pillar.lowerY:
            if (pillar.x < self.x + self.w < pillar.x + pillar.w) and self.x < pillar.x + pillar.w:
                self.fitness = self.score
                self.life = False
                self.score = 0.0
                
    def think(self, x, upperH, lowerY):
        jump = self.brain.feedForward([x, upperH, lowerY, self.x, self.y])
        if jump:
            self.y -= 14

    def reset(self):
        self.y = height/2 - 20
        self.life = True

# Population Pool
class Population:
    def __init__(self):
        self.population = []
        self.eliminated = []

    def createPopulation(self):
        colors = [(241, 148, 138), (187, 143, 206), (133, 193, 233),
                  (171, 235, 198), (249, 231, 159), (237, 187, 153),
                  (202, 207, 210), (241, 196, 15), (169, 50, 38),
                  (91, 44, 111), (33, 97, 140), (25, 111, 61)]

        for i in xrange(len(colors)):
            bird = Bird(colors[i])
            self.population.append(bird)

    def draw(self, pillar):
        for bird in self.population:
            bird.draw(pillar)

    def move(self, pillar):
        for bird in self.population:
            bird.move(pillar)

        popCopy = self.population[:]
        for bird in popCopy:
            if not bird.life:
                self.eliminated.append(bird)
                self.population.remove(bird)

        if self.population == []:
            self.evolve()

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
            with open("Trained_biases-Best/biases_" + str(index) + ".pkl", "rb") as f:
                self.population[i].brain.biases = pickle.load(f)
            with open("Trained_weights-Best/weights_" + str(index) + ".pkl", "rb") as f:
                self.population[i].brain.weights = pickle.load(f)
            index += 1

    def evolve(self):
        self.reproduce()

    def normalizeFitness(self):
        summation = 0
        for bird in self.population:
            summation += bird.fitness

        for i in xrange(len(self.population)):
            self.population[i].prob = float(self.population[i].fitness)/summation
        
    def reproduce(self):
        global pillarSet, generation, bestScoreGeneration, bestSoFar, speed

        generation += 1
 
        self.population = self.eliminated[:]
        self.eliminated = []

        resetPillars()

        self.normalizeFitness()

        index = 0
        fitness1 = self.population[indexA].fitness
        
        for i in xrange(len(self.population)):
            if self.population[i].fitness > self.population[index].fitness:
                fitness1 = self.population[i].fitness
                index = i
        
        bestScoreGeneration = fitness1

        if bestScoreGeneration > bestSoFar:
            bestSoFar = bestScoreGeneration
            save_best_generation(self.population)

        for i in xrange(len(self.population)):
            self.population[i].reset()

        newPop = Population()

        i = 0

        while len(newPop.population) < len(self.population):
            indexA = pickOne(self.population)
            indexB = pickOne(self.population)

            childA = Bird(self.population[i].color)
            i += 1
            childB = Bird(self.population[i].color)
            i += 1

            childA.brain, childB.brain = self.crossover(self.population[indexA].brain, self.population[indexB].brain)

            childA.brain = self.mutation(childA.brain)
            childB.brain = self.mutation(childB.brain)

            newPop.population.append(childA)
            newPop.population.append(childB)
            
        self.population = newPop.population[:]

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
                    if random.randint(0, 1):
                        k = random.randint(0, len(brain.weights[i][j]) - 1)
                        val = np.random.randn(1)[0]
                        brain.weights[i][j][k] = val

        return brain
    
    def displayInfo(self):
        text = font2.render("Generation: " + str(generation), True, (236, 240, 241))
        display.blit(text, (10, 10))

        text = font2.render("Speed: " + str(speed/60.0) + "x", True, (236, 240, 241))
        display.blit(text, (10, height - 90))

        text = font2.render("Best Score From Previous Generation: " + str(bestScoreGeneration), True, (236, 240, 241))
        display.blit(text, (10, height - 60))

        text = font2.render("Best From All The Generations: " + str(bestSoFar), True, (236, 240, 241))
        display.blit(text, (10, height - 30))

        i = 0
        for species in self.population:
            text = font.render("Score: " + str("{0:.4f}".format(species.score)), True, species.color)
            display.blit(text, (10, 30 + (i + 1)*15))
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

def gameLoop():
    loop = True
    global pillarSet, generation, bestScoreGeneration, bestSoFar, speed, index, pillarGap

    pillarGap = 200
    
    generation = 1
    bestScoreGeneration = 0
    bestSoFar = 0

    pillarSet = []
    noPillar = 15
    for i in xrange(noPillar):
        pillar = Pillar(width + i*pillarGap)
        pillarSet.append(pillar)

    population = Population()
    population.createPopulation()
    speed = 60

    index = 0

    while loop:
        if len(pillarSet) < noPillar:
            index = 0
            for i in xrange(len(pillarSet)):
                if pillarSet[i].x > pillarSet[index].x:
                    index = i
            pillar = Pillar(pillarSet[len(pillarSet) - 1].x + pillarGap)
            pillarSet.append(pillar)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    close()
                if event.key == pygame.K_s:
                    save_best_generation(population.population)
                if event.key == pygame.K_l:
                    population.loadData()
                if event.key == pygame.K_b:
                    population.loadBest()
                if event.key == pygame.K_r:
                    gameLoop()
                if event.key == pygame.K_1:
                    speed = 60*1
                if event.key == pygame.K_2:
                    speed = 60*2
                if event.key == pygame.K_3:
                    speed = 60*3
                if event.key == pygame.K_4:
                    speed = 60*4
                if event.key == pygame.K_5:
                    speed = 60*5
                if event.key == pygame.K_6:
                    speed = 60*6
                if event.key == pygame.K_7:
                    speed = 60*7
                if event.key == pygame.K_8:
                    speed = 60*8
                if event.key == pygame.K_9:
                    speed = 60*9

        display.fill((39, 55, 70))

        for i in xrange(len(pillarSet)):
            pillarSet[i].move()
            if pillarSet[i].x < population.population[0].x:
                index = (i + 1)%noPillar
            pillarSet[i].draw()
        
        population.move(pillarSet[index])        
        population.draw(pillarSet[index])

        population.displayInfo()

        pygame.display.update()
        clock.tick(speed)

gameLoop()
