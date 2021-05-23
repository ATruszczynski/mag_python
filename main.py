import numpy as np
from neural_network import *
from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
from sklearn.linear_model import LinearRegression

from queue import PriorityQueue

q = PriorityQueue()

q.put((4, 'Read'))
q.put((2, 'Play'))
q.put((5, 'Write'))
q.put((1, 'Code'))
q.put((3, 'Study'))

print(list(q.queue))

while not q.empty():
    next_item = q.get()
    print(next_item)