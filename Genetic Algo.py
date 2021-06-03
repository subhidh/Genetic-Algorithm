import numpy as np, random, operator, pandas as pd
import turtle
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def draw(self):
        return ([self.x, self.y])
    
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    arr=population[sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)[0][0]]
    t1.clear()
    for i in range(len(arr)):
        if(i<len(arr)-1):
            t1.penup()
            t1.goto(arr[i].draw()[0], arr[i].draw()[1])
            t1.pendown()
            t1.goto(arr[i+1].draw()[0], arr[i+1].draw()[1])
        else:
            t1.goto(arr[i].draw()[0], arr[i].draw()[1])
            t1.pendown()
            t1.goto(arr[0].draw()[0], arr[0].draw()[1])
    t2.penup()
    t2.goto(300,350)
    t2.clear()
    dist.append(1/(sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)[0][1]))
    t2.write("Current Distance = "+ str(1/(sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)[0][1])))
    
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate):
    pop = initialPopulation(popSize, population)
    turtle.penup()
    turtle.goto(300,370)
    turtle.write("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    i=1
    while True:
        pop = nextGeneration(pop, eliteSize, mutationRate)
        t3.clear()
        t3.penup()
        t3.goto(300,360)
        t3.write("Generation = " +str(i))
        i+=1
        if( i>100 and (np.max(np.array(dist[-80:])) == np.min(np.array(dist[-80:]))) ):
            break
    turtle.penup()
    turtle.goto(300,340)
    turtle.write("Final distance: " + str(1 / rankRoutes(pop)[0][1]))    
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

cityList = []
dist=[]
for i in range(0,30):
    cityList.append(City(x=int(random.random() * 300), y=int(random.random() * 300)))

screen=turtle.getscreen()
turtle.setworldcoordinates(0, 0, 500, 500)
turtle.hideturtle()
turtle.tracer(n=1, delay=0.5)
t1 = turtle.Turtle()
t1.hideturtle()
t2 = turtle.Turtle()
t2.hideturtle()
t3 = turtle.Turtle()
t3.hideturtle()
    
t1.speed('fastest')
turtle.speed('fastest')
for i in range(len(cityList)):
    turtle.pencolor("red")
    turtle.penup()
    turtle.goto(cityList[i].draw()[0], cityList[i].draw()[1])
    turtle.pendown()
    turtle.dot(5)
turtle.pencolor("black")
turtle.penup()
turtle.goto(300,380)
turtle.write("Mutation Rate = 1%")

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01)
