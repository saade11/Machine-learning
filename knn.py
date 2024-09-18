import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math



def classifyAPoint(points, p, k = 3):
    
    distance = []
    for group in points:
        for feature in points[group]:
            euclidian_distance = math.sqrt((feature[0]-p[0])**2 + (feature[1]-p[1])**2)
            distance.append((euclidian_distance,group))
    print(distance)
    distance = sorted(distance)[:k]
    print(distance)
    freq1 = 0
    freq2 = 0
    
    for d in distance:
        if d[1] == 0:
            freq1 = freq1 + 1
        elif d[1] == 1:
            freq2 = freq2 + 1
            
    if freq1 > freq2:
        return "blue"
    else:
        return "red"
    
def main():
    points = {"red":[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)],
              "blue":[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}

    p = (2.5,7)
    k = 3
    print("The value classified to unknown point is: {}".\
          format(classifyAPoint(points,p,k)))
    
if __name__ == '__main__':
    main()