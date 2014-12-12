#! /usr/local/bin/python3


from collections import defaultdict, Counter




######### Construct transition/reward functions #########

locations = defaultdict(Counter)
rewards = defaultdict(list)

with open("small.csv") as f:
  f.readline()
  for line in f:
    s1, s2, a, p1, p2, r = line.split(',')
    s1, s2, a, p1, p2, r = int(s1)-1, int(s2)-1, int(a)-1, int(p1)-1, int(p2)-1, int(r)
    locations[(s1, s2, a)][(p1, p2)] += 1
    rewards[(s1, s2, a)].append(r)
    

transitions = defaultdict(list)

for x in range(10):
  for y in range(10):
    for action in range(5):
      counter = locations[(x, y, action)]
      count = sum(counter.values())
      transitions[(x, y, action)] = [(counter[k]/count, k) for k in counter.keys()]

for i in rewards:
  rewards[i] = sum(rewards[i]) / len(rewards[i])


######### Construct utility function #########

gamma = 0.9

U = [ [(0,0) for j in range(10)] for i in range(10)]
Up = [ [(0,0) for j in range(10)] for i in range(10)]

for _ in range(100):
  U, Up = Up, U
  for x in range(10):
    for y in range(10):
      results = []
      for action in range(5):
        results.append( (action, rewards[(x, y, action)] + gamma * sum( p * Up[xp][yp][1] for p, (xp, yp) in transitions[(x, y, action)])))
      U[x][y] = max(results, key = lambda x: x[1])


######### Write policy #########

with open("small.policy", "w") as f:
  for y in range(10):
    for x in range(10):
      f.write(str(U[x][y][0]+1) + "\n")
          
