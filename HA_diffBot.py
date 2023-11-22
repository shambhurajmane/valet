import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd



class  DiffBot:       
    speedPrecision = 4
    wheelSeperation=0.287 * 5           #lets take it as grid dimension
    botDiameter=0.32 * 5 
    wheelDiameter=0.066 *5
    maxVelocity=2.0
    minVelocity=1.0
    step_size=0.5

   
class Cost:      
    reverse = 30
    directionChange = 300
    steerAngle = 10
    hybridCost = 200
    lowSpeedCost=0
    steerAngleChange = 10
    
    
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
        print("map")
        # calculate min max map grid index based on obstacles in map
        mapMinX = round(min(obstacleX) / xyResolution)
        mapMinY = round(min(obstacleY) / xyResolution)
        mapMaxX = round(max(obstacleX) / xyResolution)
        mapMaxY = round(max(obstacleY) / xyResolution)

        # create a KDTree to represent obstacles
        ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])
        print(len(obstacleX))

        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  

def index(Node):       
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([Node.gridIndex[0], Node.gridIndex[1], Node.gridIndex[2]])

def motionCommands():     


    # Motion commands for a Non-Holonomic Robot like a Differential drive robot (Trajectories using left wheel and right wheel velocity )
    
    angle=45
    step_length = DiffBot.maxVelocity / 2
    vr= step_length /(DiffBot.step_size* math.cos(math.radians(angle))) + ( DiffBot.wheelSeperation * math.tan(math.radians(angle)))/2
    vl= vr - (DiffBot.wheelSeperation * math.tan(math.radians(angle)))
    
    motionCommand = [[DiffBot.maxVelocity, DiffBot.maxVelocity],[vr,vl],[-vr,-vl]]

    return motionCommand


def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta


def holonomicMotionCommands():     

    # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
    holonomicMotionCommand = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return holonomicMotionCommand

def kinematicSimulationNode(currentNode, motionCommand, mapParameters, simulationLength=1, step = 0.2 ):      

    # Simulate node using given current Node and Motion Commands
    traj = []
    angle = pi_2_pi(currentNode.traj[-1][2] + (motionCommand[0]-motionCommand[1]) * step /  DiffBot.wheelSeperation)
    
    traj.append([currentNode.traj[-1][0] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.cos(angle),
                currentNode.traj[-1][1] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.sin(angle),
                pi_2_pi(angle + (motionCommand[0]-motionCommand[1]) * step /  DiffBot.wheelSeperation)])
    for i in range(int((simulationLength/step))-1):
        traj.append([traj[i][0] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.cos(traj[i][2]),
                    traj[i][1] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.sin(traj[i][2]),
                    # pi_2_pi((motionCommand[0]-motionCommand[1]) * step /  DiffBot.wheelSeperation)])
                    pi_2_pi(traj[i][2] + (motionCommand[0]-motionCommand[1]) * step /  DiffBot.wheelSeperation)])

    # Find grid index
    # print(traj)
    gridIndex = [round(traj[-1][0]/mapParameters.xyResolution), \
                 round(traj[-1][1]/mapParameters.xyResolution), \
                 round(traj[-1][2]/mapParameters.yawResolution)]
 
    # Check if node is valid
    if not isValid(traj, gridIndex, mapParameters):
        
        return None

    # Calculate Cost of the node
    cost = simulatedPathCost(currentNode, motionCommand, simulationLength)

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

    diffBotRadius = ( DiffBot.botDiameter)/4 + 1
    for i in traj:
        cx = i[0] 
        cy = i[1] 
        pointsInObstacle = mapParameters.ObstacleKDTree.query_ball_point([cx, cy], diffBotRadius)

        if not pointsInObstacle:
            continue

        for p in pointsInObstacle:
            xo = mapParameters.obstacleX[p] - cx
            yo = mapParameters.obstacleY[p] - cy
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])

            if abs(dx) < diffBotRadius and abs(dy) <  diffBotRadius:
                return True

    return False

def simulatedPathCost(currentNode, motionCommand, simulationLength):

    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    if motionCommand[0] > 0:
        cost += simulationLength
    elif motionCommand[0] < 0 :
        cost += simulationLength * Cost.reverse

    # Steering Angle change cost
    if currentNode != motionCommand[0] :
        cost += Cost.directionChange
        
    # Steering Angle Cost
    if motionCommand[0] != DiffBot.maxVelocity:
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

    # Parking Map
    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(0)

    # for i in range(51):
    #     obstacleX.append(0)
    #     obstacleY.append(i)

    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(50)

    # for i in range(51):
    #     obstacleX.append(50)
    #     obstacleY.append(i)

    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(40)

    # for i in range(0,20):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(29,51):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(24,30):
    #     obstacleX.append(19)
    #     obstacleY.append(i) 

    # for i in range(24,30):
    #     obstacleX.append(29)
    #     obstacleY.append(i) 

    # for i in range(20,29):
    #     obstacleX.append(i)
    #     obstacleY.append(24)

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

    # Iterate till we reach start node from goal node
    while currentNodeIndex != startNodeIndex:
        a, b, c = zip(*currentNode.traj)
        x += a[::-1] 
        y += b[::-1] 
        yaw += c[::-1]
        currentNodeIndex = currentNode.parentIndex
        currentNode = closedSet[currentNodeIndex]
    return x[::-1], y[::-1], yaw[::-1]

def run(s, g, mapParameters, plt):   

    # Compute Grid Index for start and Goal node
    sGridIndex = [round(s[0] / mapParameters.xyResolution), \
                  round(s[1] / mapParameters.xyResolution), \
                  round(s[2]/mapParameters.yawResolution)]
    gGridIndex = [round(g[0] / mapParameters.xyResolution), \
                  round(g[1] / mapParameters.xyResolution), \
                  round(g[2]/mapParameters.yawResolution)]

    # Generate all Possible motion commands to Differential drive robot
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
        # print(currentNodeIndex)

        # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
        if currentNodeIndex == index(goalNode):
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
            x,y,z =zip(*simulatedNode.traj)
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
    x, y, yaw = backtrack(startNode, goalNode, closedSet, plt)
    return x, y, yaw

def drawDiffBot(x, y, yaw, color='black'):
    theta_B = math.pi + yaw

    xB = x + DiffBot.botDiameter / 4 * np.cos(theta_B)
    yB = y + DiffBot.botDiameter / 4 * np.sin(theta_B)

    theta_BL = theta_B + math.pi / 2
    theta_BR = theta_B - math.pi / 2

    x_BL = xB + DiffBot.botDiameter / 2 * np.cos(theta_BL)        # Bottom-Left vertex
    y_BL = yB + DiffBot.botDiameter / 2 * np.sin(theta_BL)
    x_BR = xB + DiffBot.botDiameter / 2 * np.cos(theta_BR)        # Bottom-Right vertex
    y_BR = yB + DiffBot.botDiameter / 2 * np.sin(theta_BR)

    x_FL = x_BL + DiffBot.botDiameter * np.cos(yaw)               # Front-Left vertex
    y_FL = y_BL + DiffBot.botDiameter * np.sin(yaw)
    x_FR = x_BR + DiffBot.botDiameter * np.cos(yaw)               # Front-Right vertex
    y_FR = y_BR + DiffBot.botDiameter * np.sin(yaw)

    plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                [y_BL, y_BR, y_FR, y_FL, y_BL],
                linewidth=1, color='black')

def main():

    # Set Start, Goal x, y, theta
    s = [10, 10, np.deg2rad(0)]
    g = [32, 33, np.deg2rad(0)]


    # Get Obstacle Map
    obstacleX, obstacleY = map()

    # Calculate map Paramaters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 1, np.deg2rad(15.0))

    # Run Hybrid A*
    x, y, yaw = run(s, g, mapParameters, plt)
    print(len(x))


    # Draw Animated Differential drive robot
    for k in range(len(x)):
        plt.cla()
        plt.xlim(min(obstacleX), max(obstacleX)) 
        plt.ylim(min(obstacleY), max(obstacleY))
        plt.plot(obstacleX, obstacleY, "sk")
        plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
        
        drawDiffBot(x[k], y[k], yaw[k])
        plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
        plt.title("Hybrid A*")
        plt.pause(0.001)
    
    plt.show()

if __name__ == '__main__':
    main()