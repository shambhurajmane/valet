import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd


class CarWithTrailer:
    maxSteerAngle = 0.6
    steerPresion = 10
    wheelBase = 2.58    # [m] wheel base: rear to front steer
    axleToFront = 3.0   # [m] distance from rear to vehicle front end of vehicle
    axleToHitch = 0.4   
    hitchToTrailer = 2.0
    axleToBack = 0.4    # [m] distance from rear to vehicle back end of vehicle
    width = 2.0   # [m] width of vehicle
    RTF = 0.4  # [m] distance from rear to vehicle front end of trailer
    RTB = 4.0  # [m] distance from rear to vehicle back end of trailer

    wheelSeperation = 0.7 * width  # [m] distance between left-right wheels

    RTR = 8.0  # [m] rear to trailer wheel
    
class Cost:
    reverse = 8
    directionChange = 100
    steerAngle = 1
    steerAngleChange = 5
    hybridCost = 50

class Node:
    def __init__(self, gridIndex, traj, steeringAngle, direction, cost, parentIndex):
        self.gridIndex = gridIndex         # grid block x, y, yaw index
        self.traj = traj                   # trajectory x, y  of a simulated node
        self.steeringAngle = steeringAngle # steering angle throughout the trajectory
        self.direction = direction         # direction throughout the trajectory
        self.cost = cost                   # node cost
        self.parentIndex = parentIndex     # parent node index

class HolonomicNode:
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex
        self.cost = cost
        self.parentIndex = parentIndex

class MapParameters:
    def __init__(self, mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY):
        self.mapMinX = mapMinX               # map min x coordinate(0)
        self.mapMinY = mapMinY               # map min y coordinate(0)
        self.mapMaxX = mapMaxX               # map max x coordinate
        self.mapMaxY = mapMaxY               # map max y coordinate
        self.xyResolution = xyResolution     # grid block length
        self.yawResolution = yawResolution   # grid block possible yaws
        self.ObstacleKDTree = ObstacleKDTree # KDTree representating obstacles
        self.obstacleX = obstacleX           # Obstacle x coordinate list
        self.obstacleY = obstacleY           # Obstacle y coordinate list

def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):
        
        # calculate min max map grid index based on obstacles in map
        mapMinX = round(min(obstacleX) / xyResolution)
        mapMinY = round(min(obstacleY) / xyResolution)
        mapMaxX = round(max(obstacleX) / xyResolution)
        mapMaxY = round(max(obstacleY) / xyResolution)

        # create a KDTree to represent obstacles
        ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])

        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  

def index(Node):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([Node.gridIndex[0], Node.gridIndex[1], Node.gridIndex[2], Node.gridIndex[3]])

def motionCommands():

    # Motion commands for a Non-Holonomic Robot like a CarWithTrailer or Bicycle (Trajectories using Steer Angle and Direction)
    direction = 1
    motionCommand = []
    for i in np.arange(CarWithTrailer.maxSteerAngle, -(CarWithTrailer.maxSteerAngle), -CarWithTrailer.maxSteerAngle/CarWithTrailer.steerPresion):
        motionCommand.append([i, direction])
        motionCommand.append([i, -direction])
    return motionCommand

def holonomicMotionCommands():

    # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
    holonomicMotionCommand = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return holonomicMotionCommand

def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta

def kinematicSimulationNode(currentNode, motionCommand, mapParameters, simulationLength=3, step = 0.6 ):

    # Simulate node using given current Node and Motion Commands
    traj = []
    L1 = CarWithTrailer.axleToHitch
    L2= CarWithTrailer.hitchToTrailer
    L = CarWithTrailer.wheelBase
    
    angleC = pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * step / CarWithTrailer.wheelBase * math.tan(motionCommand[0]))
    angleT = pi_2_pi(currentNode.traj[-1][3] + step * motionCommand[1] / CarWithTrailer.hitchToTrailer * math.sin(currentNode.traj[-1][2] - currentNode.traj[-1][3]))

    
    traj.append([currentNode.traj[-1][0] + motionCommand[1] * step * math.cos(angleC),
                currentNode.traj[-1][1] + motionCommand[1] * step * math.sin(angleC),
                pi_2_pi(angleC + motionCommand[1] * step / CarWithTrailer.wheelBase * math.tan(motionCommand[0])),
                pi_2_pi(angleT + step * motionCommand[1] / CarWithTrailer.hitchToTrailer * math.sin(angleC - angleT))])
    for i in range(int((simulationLength/step))-1):
        traj.append([traj[i][0] + motionCommand[1] * step * math.cos(traj[i][2]),
                    traj[i][1] + motionCommand[1] * step * math.sin(traj[i][2]),
                    pi_2_pi(traj[i][2] + motionCommand[1] * step / CarWithTrailer.wheelBase * math.tan(motionCommand[0])),
                    pi_2_pi(traj[i][3] + step * motionCommand[1] / CarWithTrailer.hitchToTrailer * math.sin(traj[i][2] - traj[i][3]))])
    
    # Find grid index
    gridIndex = [round(traj[-1][0]/mapParameters.xyResolution), \
                 round(traj[-1][1]/mapParameters.xyResolution), \
                 round(traj[-1][2]/mapParameters.yawResolution), \
                 round(traj[-1][3]/mapParameters.yawResolution)]
    # print(traj, gridIndex, mapParameters)
    # Check if node is valid
    if not isValid(traj, gridIndex, mapParameters):
        
        return None

    # Calculate Cost of the node
    cost = simulatedPathCost(currentNode, motionCommand, simulationLength)
    print(traj)

    return Node(gridIndex, traj, motionCommand[0], motionCommand[1], cost, index(currentNode))


def isValid(traj, gridIndex, mapParameters):

    # Check if Node is out of map bounds
    if gridIndex[0]<=mapParameters.mapMinX or gridIndex[0]>=mapParameters.mapMaxX or \
       gridIndex[1]<=mapParameters.mapMinY or gridIndex[1]>=mapParameters.mapMaxY:
        return False

    # Check if Node is colliding with an obstacle
    if collision(traj, mapParameters):
        # print("not valid")
        return False
    return True

def collision(traj, mapParameters):

    carRadius = (CarWithTrailer.axleToFront + CarWithTrailer.axleToHitch)/2 + 1
    trailerRadius = (CarWithTrailer.RTB + CarWithTrailer.hitchToTrailer)/2 + 1
    dl = (CarWithTrailer.axleToFront - CarWithTrailer.axleToHitch)/2
    
    for i in traj:
        cx = i[0] + dl * math.cos(i[2])
        cy = i[1] + dl * math.sin(i[2])
        
        tx = i[0] - CarWithTrailer.axleToHitch * math.cos(i[2]) - CarWithTrailer.RTB /2 * math.cos(i[3])
        ty = i[1] - CarWithTrailer.axleToHitch * math.sin(i[2]) - CarWithTrailer.RTB /2 * math.sin(i[3])
        
        pointsInObstacleCar = mapParameters.ObstacleKDTree.query_ball_point([cx, cy], carRadius)

        if not pointsInObstacleCar:
            continue

        for p in pointsInObstacleCar:
            xo = mapParameters.obstacleX[p] - cx
            yo = mapParameters.obstacleY[p] - cy
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])

            if abs(dx) < carRadius and abs(dy) < CarWithTrailer.width / 2 + 1:
                return True
        
        pointsInObstacleTrailer = mapParameters.ObstacleKDTree.query_ball_point([tx, ty], trailerRadius)

        if not pointsInObstacleTrailer:
            continue

        for p in pointsInObstacleTrailer:
            xo = mapParameters.obstacleX[p] - tx
            yo = mapParameters.obstacleY[p] - ty
            dx = xo * math.cos(i[3]) + yo * math.sin(i[3])
            dy = -xo * math.sin(i[3]) + yo * math.cos(i[3])

            if abs(dx) < trailerRadius and abs(dy) < CarWithTrailer.width / 2 + 1:
                return True

    return False

def simulatedPathCost(currentNode, motionCommand, simulationLength):

    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    if motionCommand[1] == 1:
        cost += simulationLength 
    else:
        cost += simulationLength * Cost.reverse

    # Direction change cost
    if currentNode.direction != motionCommand[1]:
        cost += Cost.directionChange

    # Steering Angle Cost
    cost += motionCommand[0] * Cost.steerAngle

    # Steering Angle change cost
    cost += abs(motionCommand[0] - currentNode.steeringAngle) * Cost.steerAngleChange

    return cost

def eucledianCost(holonomicMotionCommand):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

def holonomicNodeIndex(HolonomicNode):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([HolonomicNode.gridIndex[0], HolonomicNode.gridIndex[1]])

def obstaclesMap(obstacleX, obstacleY, xyResolution):

    # Compute Grid Index for obstacles
    obstacleX = [round(x / xyResolution) for x in obstacleX]
    obstacleY = [round(y / xyResolution) for y in obstacleY]

    # Set all Grid locations to No Obstacle
    obstacles =[[False for i in range(max(obstacleY))]for i in range(max(obstacleX))]

    # Set Grid Locations with obstacles to True

    for i, j in zip(obstacleX, obstacleY): 
        obstacles[i][j] = True
        break

    return obstacles

def holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):

    # Check if Node is out of map bounds
    if neighbourNode.gridIndex[0]<= mapParameters.mapMinX or \
       neighbourNode.gridIndex[0]>= mapParameters.mapMaxX or \
       neighbourNode.gridIndex[1]<= mapParameters.mapMinY or \
       neighbourNode.gridIndex[1]>= mapParameters.mapMaxY:
        return False

    # Check if Node on obstacle
    if obstacles[neighbourNode.gridIndex[0]][neighbourNode.gridIndex[1]]:
        return False

    return True

def holonomicCostsWithObstacles(goalNode, mapParameters):

    gridIndex = [round(goalNode.traj[-1][0]/mapParameters.xyResolution), round(goalNode.traj[-1][1]/mapParameters.xyResolution)]
    gNode =HolonomicNode(gridIndex, 0, tuple(gridIndex))

    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)

    holonomicMotionCommand = holonomicMotionCommands()

    openSet = {holonomicNodeIndex(gNode): gNode}
    closedSet = {}

    priorityQueue =[]
    heapq.heappush(priorityQueue, (gNode.cost, holonomicNodeIndex(gNode)))

    while True:
        if not openSet:
            break

        _, currentNodeIndex = heapq.heappop(priorityQueue)
        currentNode = openSet[currentNodeIndex]
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        for i in range(len(holonomicMotionCommand)):
            neighbourNode = HolonomicNode([currentNode.gridIndex[0] + holonomicMotionCommand[i][0],\
                                      currentNode.gridIndex[1] + holonomicMotionCommand[i][1]],\
                                      currentNode.cost + eucledianCost(holonomicMotionCommand[i]), currentNodeIndex)

            if not holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):
                continue

            neighbourNodeIndex = holonomicNodeIndex(neighbourNode)

            if neighbourNodeIndex not in closedSet:            
                if neighbourNodeIndex in openSet:
                    if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                        openSet[neighbourNodeIndex].cost = neighbourNode.cost
                        openSet[neighbourNodeIndex].parentIndex = neighbourNode.parentIndex
                        # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                else:
                    openSet[neighbourNodeIndex] = neighbourNode
                    heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))

    holonomicCost = [[np.inf for i in range(max(mapParameters.obstacleY))]for i in range(max(mapParameters.obstacleX))]

    for nodes in closedSet.values():
        holonomicCost[nodes.gridIndex[0]][nodes.gridIndex[1]]=nodes.cost

    return holonomicCost

def map():
    # Build Map
    obstacleX, obstacleY = [], []

    for i in range(51):
        obstacleX.append(i)
        obstacleY.append(0)

    for i in range(51):
        obstacleX.append(0)
        obstacleY.append(i)

    for i in range(51):
        obstacleX.append(i)
        obstacleY.append(50)

    for i in range(51):
        obstacleX.append(50)
        obstacleY.append(i)
    
    for i in range(10,20):
        obstacleX.append(i)
        obstacleY.append(30) 

    for i in range(30,51):
        obstacleX.append(i)
        obstacleY.append(30) 

    for i in range(0,31):
        obstacleX.append(20)
        obstacleY.append(i) 

    for i in range(0,31):
        obstacleX.append(30)
        obstacleY.append(i) 

    for i in range(40,50):
        obstacleX.append(15)
        obstacleY.append(i)

    for i in range(25,40):
        obstacleX.append(i)
        obstacleY.append(35)

    return obstacleX, obstacleY

def backtrack(startNode, goalNode, closedSet, plt):

    # Goal Node data
    # print(closedSet[-1])
    startNodeIndex= index(startNode)
    currentNodeIndex = list(closedSet)[-1]#goalNode.parentIndex
    currentNode = closedSet[currentNodeIndex]
    x=[]
    y=[]
    yaw=[]
    yawT= []

    # Iterate till we reach start node from goal node
    while currentNodeIndex != startNodeIndex:
        a, b, c, d = zip(*currentNode.traj)
        x += a[::-1] 
        y += b[::-1] 
        yaw += c[::-1]
        yawT += d[::-1]
        currentNodeIndex = currentNode.parentIndex
        currentNode = closedSet[currentNodeIndex]
    return x[::-1], y[::-1], yaw[::-1], yawT[::-1]


def run(s, g, mapParameters, plt):

    # Compute Grid Index for start and Goal node
    sGridIndex = [round(s[0] / mapParameters.xyResolution), \
                  round(s[1] / mapParameters.xyResolution), \
                  round(s[2]/mapParameters.yawResolution), \
                  round(s[3]/mapParameters.yawResolution)]
    gGridIndex = [round(g[0] / mapParameters.xyResolution), \
                  round(g[1] / mapParameters.xyResolution), \
                  round(g[2]/mapParameters.yawResolution), \
                  round(g[3]/mapParameters.yawResolution)]

    # Generate all Possible motion commands to car
    motionCommand = motionCommands()

    # Create start and end Node
    startNode = Node(sGridIndex, [s], 0, 1, 0 , tuple(sGridIndex))
    goalNode = Node(gGridIndex, [g], 0, 1, 0, tuple(gGridIndex))

    # Find Holonomic Heuristric
    holonomicHeuristics = holonomicCostsWithObstacles(goalNode, mapParameters)

    # Add start node to open Set
    openSet = {index(startNode):startNode}
    
    closedSet = {}

    # Create a priority queue for acquiring nodes based on their cost's
    costQueue = heapdict()

    # Add start mode into priority queue
    costQueue[index(startNode)] = max(startNode.cost , Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]])
    counter = 0

    # Run loop while path is found or open set is empty
    while True:
        counter +=1
        # Check if openSet is empty, if empty no solution available
        
        if not openSet:
            print("path not found")
            break

        # Get first node in the priority queue
        currentNodeIndex = costQueue.popitem()[0]
        currentNode = openSet[currentNodeIndex]

        # Revove currentNode from openSet and add it to closedSet
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode


        # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
        if currentNodeIndex[0] == index(goalNode)[0] and currentNodeIndex[1] == index(goalNode)[1] and currentNodeIndex[2] == index(goalNode)[2] :
            print("Path Found")
            print(currentNode.traj[-1])
            break
        
        # Get all simulated Nodes from current node
        for i in range(len(motionCommand)):
            simulatedNode = kinematicSimulationNode(currentNode, motionCommand[i], mapParameters)
            
            # Check if path is within map bounds and is collision free
            if not simulatedNode:
                # print("ss")
                continue

            # Draw Simulated Node
            x,y,z,t =zip(*simulatedNode.traj)
            plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            simulatedNodeIndex = index(simulatedNode)
            
            if simulatedNodeIndex not in closedSet: 
                
                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulatedNodeIndex not in openSet:
                    openSet[simulatedNodeIndex] = simulatedNode
                    costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
                else:
                    if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
                        openSet[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
    
    # Backtrack
    # print((startNode, goalNode, closedSet, plt))
    x, y, yaw, yawT = backtrack(startNode, goalNode, closedSet, plt)

    return x, y, yaw, yawT

def drawCarWithTrailer(x, y, yaw, yawT, color='black'):
    car = np.array([[-CarWithTrailer.axleToBack, -CarWithTrailer.axleToBack, CarWithTrailer.axleToFront, CarWithTrailer.axleToFront, -CarWithTrailer.axleToBack],
                    [CarWithTrailer.width / 2, -CarWithTrailer.width / 2, -CarWithTrailer.width / 2, CarWithTrailer.width / 2, CarWithTrailer.width / 2]])

    trail = np.array([[-CarWithTrailer.RTB, -CarWithTrailer.RTB, CarWithTrailer.RTF, CarWithTrailer.RTF, -CarWithTrailer.RTB],
                      [CarWithTrailer.width / 2, -CarWithTrailer.width / 2, -CarWithTrailer.width / 2, CarWithTrailer.width / 2, CarWithTrailer.width / 2]])


    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])


    Rot3 = np.array([[math.cos(yawT), -math.sin(yawT)],
                     [math.sin(yawT), math.cos(yawT)]])

    car = np.dot(Rot1, car)


    trail = np.dot(Rot3, trail)


    car += np.array([[x], [y]])
    trail += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(trail[0, :], trail[1, :], color)


def main():

    # Set Start, Goal x, y, theta
    s = [10, 10, np.deg2rad(90), np.deg2rad(90)]
    g = [38, 32, np.deg2rad(180), np.deg2rad(180)]
    # s = [10, 35, np.deg2rad(0)]
    # g = [22, 28, np.deg2rad(0)]

    # Get Obstacle Map
    obstacleX, obstacleY = map()

    # Calculate map Paramaters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 4, np.deg2rad(15.0))

    # Run Hybrid A*
    x, y, yaw, yawT= run(s, g, mapParameters, plt)

    # Draw Animated CarWithTrailer
    for k in range(len(x)):
        # plt.cla()
        plt.xlim(min(obstacleX), max(obstacleX)) 
        plt.ylim(min(obstacleY), max(obstacleY))
        plt.plot(obstacleX, obstacleY, "sk")
        plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
        drawCarWithTrailer(x[k], y[k], yaw[k], yawT[k] )
        plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
        plt.title("Hybrid A*")
        plt.pause(0.001)
    
    plt.show()

if __name__ == '__main__':
    main()