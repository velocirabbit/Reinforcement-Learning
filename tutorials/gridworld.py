import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from skimage.transform import resize

class GameObj:
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class GameEnv:
    def __init__(self, size, n_goals, n_fire, partial = 0, img_sz = 84):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.n_goals = n_goals
        self.n_fire = n_fire
        self.img_sz = img_sz
        a = self.reset()
        plt.imshow(a, interpolation = "nearest")
        
    def reset(self):
        self.objects = []
        # Initialize game objects and add them to objects array
        hero  = GameObj(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        for _ in range(self.n_goals):
            goal = GameObj(self.newPosition(), 1, 1, 1, 1, 'goal')
            self.objects.append(goal)
        for _ in range(self.n_fire):
            fire = GameObj(self.newPosition(), 1, 1, 0, -1, 'fire')
            self.objects.append(fire)
        # Get and return the game state
        self.state = self.renderEnv()
        return self.state
        
    def moveChar(self, direction):
        # 0: up, 1: down, 2: left, 3: right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if   direction == 0 and hero.y >= 1:              # Up
            hero.y -= 1
        elif direction == 1 and hero.y <= self.sizeY-2:   # Down
            hero.y += 1
        elif direction == 2 and hero.x >= 1:              # Left
            hero.x -= 1
        elif direction == 3 and hero.x <= self.sizeX-2:   # Right
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:   # Hero didn't move
            penalize = 0.0
        self.objects[0] = hero
        return penalize
        
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        currentPositions = []
        for obj in self.objects:
            if (obj.x, obj.y) not in currentPositions:
                currentPositions.append((obj.x, obj.y))
        points = [
            xy for xy in itertools.product(*iterables) if xy not in currentPositions
        ]
        location = np.random.choice(range(len(points)), replace = False)
        return points[location]
        
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:  # Object collision! Game over
                self.objects.remove(other)
                if other.reward == 1:   # Reached goal
                    self.objects.append(
                        GameObj(self.newPosition(), 1, 1, 1, 1, 'goal')
                    )
                else:                   # Hit fire
                    self.objects.append(
                        GameObj(self.newPosition(), 1, 1, 0, -1, 'fire')
                    )
                return other.reward, False
        # Hero block hasn't hit anything else; game not over
        return 0, False
        
    def renderEnv(self):
        a = np.ones([self.sizeY+2, self.sizeX+2, 3])  # +2 for a border
        a[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial > 0:  # Only render the environment around the hero
            r = self.partial
            minX = min(max(0, hero.x+1-r), self.sizeX+1-2*r)
            minY = min(max(0, hero.y+1-r), self.sizeY+1-2*r)
            maxX = max(min(self.sizeX+2, hero.x+r+2), 2*r+1)
            maxY = max(min(self.sizeY+2, hero.y+r+2), 2*r+1)
            a = a[minY:maxY, minX:maxX, :]
        # Resize the rendered image
        r = resize(a[:,:,0], [self.img_sz, self.img_sz], order = 0)
        g = resize(a[:,:,1], [self.img_sz, self.img_sz], order = 0)
        b = resize(a[:,:,2], [self.img_sz, self.img_sz], order = 0)
        a = np.stack([r, g, b], axis = 2)
        return a
        
    def step(self, action):
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done, reward, penalty, sep = '\n')
        return state, (reward + penalty), done
        
