using Graphs
using BayesNets
import BayesNets.prior, BayesNets.logBayesScore
using DataFrames

# By default, do 10 random restart hill climbs.
RUNS = 10
ANNEAL = false


infile = ARGS[1]
data = readtable(ARGS[1])
nodes = names(data)



################## Local score calculation functions ##################

# compute the contribution of a single node to the overall Bayes score
function logBayesScoreNode(i, b::BayesNet, d::Matrix{Int}, parents)
  alphai, Ni = nodestats(b, i, 0, d, parents)
  p = 0.
  if !isempty(Ni)
    p += sum(lgamma(alphai + Ni))
    p -= sum(lgamma(alphai))
    p += sum(lgamma(sum(alphai,1)))
    p -= sum(lgamma(sum(alphai,1) + sum(Ni,1)))
  end
  p
end


# replacement for the statistics functions in learning.jl
# gathers statistics relevant to specified node
function nodestats(b, i, alpha, d::Matrix{Int}, parents_)
  # the N = statistics(b) part
  r = [length(domain(b, node).elements) for node in b.names]
  parents = collect(parents_)
  
  q = 1
  if !isempty(parents)
    q = prod(r[parents])
  end
  Ni_notalphad = ones(r[i], q) # equivalent to prior(b)[i]

  Ni = Ni_notalphad * alpha 

  # the statistics!(N, b, d) part
  rp = r[parents]
  (n, m) = size(d)
  for di = 1:m
    k = d[i,di]
    j = 1
    if !isempty(parents)
      ndims = length(parents)
      index = int(d[parents[1],di])
      stride = 1
      for kk=2:ndims
        stride = stride * rp[kk-1]
        index += (int(d[parents[kk], di])-1) * stride
      end
      j = index
    end
    Ni[k,j] += 1.
  end
  Ni_notalphad, Ni
end





################## Output helper ##################

function output(bn::BayesNet, filename)
  f = open(filename, "w")
  for edge in edges(bn.dag)
    @printf(f, "%s, %s\n", nodes[source(edge)], nodes[target(edge)])
  end
  close(f)
end



################## Keep track of parents to save time (maybe) ##################

function myAddEdgeIndexed!(bn::BayesNet, parentkeeper, x, y)
  push!(parentkeeper[y], x)
  addEdge!(bn, nodes[x], nodes[y])
end

function myRemoveEdgeIndexed!(bn::BayesNet, parentkeeper, x, y)
  pop!(parentkeeper[y], x)
  removeEdge!(bn, nodes[x], nodes[y])
end



################## Actual implementation ##################


# for simplicity, this handles both regular climbing and annealing
function acceptablechange(oldscore, newscore, temperature, anneal)
  if newscore > oldscore
    return true
  else
    if !anneal
      return false
    else
      return exp((newscore - oldscore) / temperature) > rand()
    end
  end
end

# can do regular climbing and annealing
function climb!(bn::BayesNet, datamat::Array{Int64, 2}, anneal = false) 
  parentkeeper = [Set{Int64}() for i in 1:length(bn.names)]
  scorekeeper = [logBayesScoreNode(i, bn, datamat, parentkeeper[i]) for i in 1:length(bn.names)]
  
  temperature = 10000
  coolingfactor = .9
  touched = true
  while (!anneal && touched) || (anneal && temperature > 100)
    temperature *= coolingfactor
    touched = false
    shuffled1 = randperm(length(bn.dag.vertices))
    shuffled2 = randperm(length(bn.dag.vertices))
    
    # loop over all possible edges (in random order)
    for xn in bn.dag.vertices, yn in bn.dag.vertices
      x, y = shuffled1[xn], shuffled2[yn]
      if x == y
        continue
      end

      if in(x, parentkeeper[y]) # ie, edge exists
        # try removing and reversing
        curycontrib = scorekeeper[y]
        curxcontrib = scorekeeper[x]
        
        myRemoveEdgeIndexed!(bn, parentkeeper, x, y)        
        afterycontrib = logBayesScoreNode(y, bn, datamat, parentkeeper[y])
        
        myAddEdgeIndexed!(bn, parentkeeper, y, x)
        if !isValid(bn)
          myRemoveEdgeIndexed!(bn, parentkeeper, y, x)
          if !acceptablechange(curycontrib, afterycontrib, temperature, anneal)
            myAddEdgeIndexed!(bn, parentkeeper, x, y)
            continue
          else
            scorekeeper[y] = afterycontrib
          end
        else
          #both removing and reversing are valid
          afterxcontrib = logBayesScoreNode(x, bn, datamat, parentkeeper[x])
                    
          deltarem = afterycontrib - curycontrib
          deltaadd = afterxcontrib - curxcontrib + afterycontrib - curycontrib

          candidate = max(deltarem, deltaadd)
          
          if candidate < 0
            myRemoveEdgeIndexed!(bn, parentkeeper, y, x)
            myAddEdgeIndexed!(bn, parentkeeper, x, y)
            continue
          elseif candidate == deltarem
            myRemoveEdgeIndexed!(bn, parentkeeper, y, x)
            scorekeeper[y] = afterycontrib
          else
            scorekeeper[y] = afterycontrib
            scorekeeper[x] = afterxcontrib
          end
        end
      else
        # try adding
        curycontrib = scorekeeper[y]
        myAddEdgeIndexed!(bn, parentkeeper, x, y)
        afterycontrib = logBayesScoreNode(y, bn, datamat, parentkeeper[y])
        
        if (!isValid(bn)) || afterycontrib <= curycontrib
          myRemoveEdgeIndexed!(bn, parentkeeper, x, y)
          continue
        end
        scorekeeper[y] = afterycontrib
      end
      touched = true
    end
  end
  
  logBayesScore(bn, data)
end


function climbRandRestart(data, anneal = false, runs = 10)
  best, bestseed = -Inf, 0
  for runnum in 1:runs
    # Need to reset the net each time. This is the easiest way.
    bn = BayesNet(nodes)
    bn.domains = [DiscreteDomain([x for x in unique(data[label])]) for label in names(data)];
    datamat = indexData(bn, data)
    
    srand(0xcafe*runnum)
    
    score = climb!(bn, datamat, anneal)

    if score > best
      output(bn, replace(infile, "csv", "gph"))
      best, bestseed = score, runnum
    end
  end
  println("Best seed: 0xcafe*$bestseed")
  println("Best score: $best")
end



climbRandRestart(data, ANNEAL, RUNS)
