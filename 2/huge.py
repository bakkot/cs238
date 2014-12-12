#! /usr/local/bin/python3


from collections import defaultdict, Counter
from random import random




######### Construct transition functions #########

def base3(n):
  return ((n == 0) and '0') or (base3(n // 3).lstrip('0') + str(n % 3)) 

def tri(n):
  return "{0:0>4}".format( base3(n) )

locations1 = defaultdict(Counter)
locations2 = defaultdict(Counter)
locations3 = defaultdict(Counter)
locations4 = defaultdict(Counter)

with open("huge.csv") as f:
  f.readline()
  for line in f:
    s, a, p, r = line.split(',')
    s = "{0:0>7}".format(str(int(s) - 1))
    a = tri(int(a) - 1)
    p = "{0:0>7}".format(str(int(p) - 1))
    s1, s2, s3, s4 = s[:1], s[1:3], s[3:5], s[5:]
    a1, a2, a3, a4 = a
    p1, p2, p3, p4 = p[:1], p[1:3], p[3:5], p[5:]
    locations1[(s1, a1)][p1] += 1
    locations2[(s2, a2)][p2] += 1
    locations3[(s3, a3)][p3] += 1
    locations4[(s4, a4)][p4] += 1
    

######### Reward function #########

def R(state, action):
  if state == 6 and action == 2:
    return 100
  else:
    return 0


######### Construct utility functions #########

components = [locations1, locations2, locations3, locations4]
gamma = 0.90

Us = [[] for i in range(4)]
for i, component in enumerate(components):
  states = 10 if i==0 else 100
  formatstr = "{}" if i==0 else "{0:0>2}"
  U = [(0,0) for j in range(states)]
  Up = [(0,0) for j in range(states)]
  
  for _ in range(1000):
    U, Up = Up, U
    for s in range(states):
      ps = []
      for action in range(3):
        counter = component[(formatstr.format(str(s)), str(action))]
        count = sum(counter.values())
        ps.append( [(counter[k]/count, k) for k in counter.keys()] )
      a0r = R(s, 0) + gamma * sum( p * Up[int(sp)][1] for p, sp in ps[0] )
      a1r = R(s, 1) + gamma * sum( p * Up[int(sp)][1] for p, sp in ps[1] )
      a2r = R(s, 2) + gamma * sum( p * Up[int(sp)][1] for p, sp in ps[2] )
      U[s] = max((0, a0r), (1, a1r), (2, a2r), key=lambda x: x[1])
  Us[i] = U


######### Write policy #########

with open("huge.policy", "w") as f:
  for x in range(10):
    for y in range(100):
      for z in range(100):
        for w in range(100):
          action = 27 * Us[0][x][0] + 9 * Us[1][y][0] + 3 * Us[2][z][0] + 1 * Us[3][w][0] + 1
          f.write(str(action) + "\n")
          

######### Test #########

state = 20
component = components[1]
reward = 0
for x in range(2000):
  action = Us[1][state][0]
  counter = component[(formatstr.format(str(state)), str(action))]
  count = sum(counter.values())
  ps = [(counter[k]/count, k) for k in counter.keys()]
  for i, pr in enumerate(ps):
    p, r = pr
    p += 0 if i==0 else ps[i-1][0]
    ps[i] = p,r
  reward += R(state, action)  
  res = random()
  for p, r in ps:
    if res < p:
      state = int(r)
      break
  else:
    print(res)
    print(ps)

print(reward)
