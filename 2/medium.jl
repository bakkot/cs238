data, header = readcsv("medium.csv", header=true)



tickspersec = 100
M, m, l, g = 3.0, 1.0, 1.0, 9.8
steps = 10



thetas = sort(unique(data[:,1]))
omegas = sort(unique(data[:,2]))

# Convert a (discretized) theta or omega value to its index
th2ind = Dict{Float64, Int64}()
for (i, v) in enumerate(thetas)
  th2ind[v] = i
end

om2ind = Dict{Float64, Int64}()
for (i, v) in enumerate(omegas)
  om2ind[v] = i
end




######### Simulate the system, used in constructing transition function #########

# Discretization for theta, omega
function thfix(theta)
  l = searchsortedlast(thetas, (2*pi + theta) % (2*pi))
  
  if abs(thetas[(l % length(thetas)) + 1] - theta) < abs(thetas[((l+500) % length(thetas)) + 1] - theta)
    return thetas[(l % length(thetas)) + 1]
  else
    return thetas[((l+500) % length(thetas)) + 1]
  end
end

function omfix(omega)
  l = searchsortedlast(omegas, omega)
  
  if l == 501 
    return omegas[501]
  elseif l == 0
    return omegas[1]
  end
  
  if abs(omegas[l] - omega) < abs(omegas[l+1]-omega)
    return omegas[l]
  else
    return omegas[l+1]
  end
end


function afix(x)
  return (2*pi + x) % (2*pi)
end


# I can't wait for @inline to be included. In the mean time, macro.
# equation from https://en.wikipedia.org/wiki/Inverted_pendulum#Inverted_pendulum_on_a_cart
# and http://www.wolframalpha.com/input/?i=solve+for+x+and+y+in+%28M%2Bm%29x+-+m*l*y*cos%28t%29+%2B+m*l*u%5E2+*+sin%28t%29+%3D+F%3B+l*y+-+g*sin%28t%29+%3D+x+*+cos%28t%29
macro ff2(ft, ftp, F)
  return :((-M * g * sin($ft) - F * cos($ft) - g * m * sin($ft) + l * m * $ftp^2 * sin($ft) * cos($ft)) / (l*(-M + m * cos($ft)^2 - m)))
end

function tick(theta::Float64, omega::Float64, F::Float64)
  omegaprime = @ff2(theta, omega, F)
  
  newtheta = afix(theta + omega/tickspersec)
  newomega = omega + omegaprime/tickspersec
  
  return newtheta, newomega
end

function multitick(steps, theta, omega, F)
  for i in 1:steps
    theta, omega = tick(theta, omega, F)
  end
  return thfix(theta), omfix(omega)
end



######### Construct transition function #########

T = Array((Float64,Float64), 501,501,2)

# Via simulation
for i in 1:501, j in 1:501, k in 1:2
  T[i, j, k] = multitick(steps, thetas[i], omegas[j], 10.0*(k-1))
end



######### Reward function #########

function R(theta, omega, F)
  if theta == 0 && omega == 0
    return F == 0 ? 100 : -100
  else
    return 0.0
  end
end



######### Construct utility function #########

U = zeros(501,501)
Up = zeros(501,501)

gamma = 0.99

for x in 1:100
  println(x)
  U, Up = Up, U # Only maintain 'old' and 'new' utilities, rather than one for each state. U is the one getting updated.
  for (i, theta) in enumerate(thetas)
    for (j, omega) in enumerate(omegas)
      theta1, omega1 = T[i, j, 1]
      theta2, omega2 = T[i, j, 2]

      th1, om1 = th2ind[theta1], om2ind[omega1]
      th2, om2 = th2ind[theta2], om2ind[omega2]
      
      # Calculate rewards for both possible actions.
      # Almost all of the complexity here is dealing with noise - the actually important lines are marked.
      # 0 force case
      othercount1 = 0
      othercontrib1 = 0.0
      for toff in -1:1
        if (th1 == 1 || th1 == 501) && toff != 0
          continue # no noise in theta when entering theta=0
        end
        for omoff in -1:1
          if (om1 == 251) && omoff != 0
            continue # no noise in omega when entering omega=0
          end
          if om1+omoff >= 1 && om1+omoff <= 501
            othercount1 += 1
            othercontrib1 += Up[(((th1+toff+500)%501)+1), om1+omoff] # Utility from this possible outcome
          end
        end
      end
      fract = gamma * 1/othercount1 # utilities are weighted by gamma and likelihood (for this problem, uniform over possibilities)
      a1r = R(theta, omega, 0) + fract * othercontrib1 # utility of this action
      
      # 10 force case
      othercount2 = 0
      othercontrib2 = 0.0
      for toff in -1:1
        if (th2 == 1 || th2 == 501) && toff != 0
          continue # no noise in theta when entering theta=0
        end
        for omoff in -1:1
          if (om2 == 251) && omoff != 0
            continue # no noise in omega when entering omega=0
          end
          if om2+omoff >= 1 && om2+omoff <= 501
            othercount2 += 1
            othercontrib2 += Up[(((th2+toff+500)%501)+1), om2+omoff] # Utility from this possible outcome
          end
        end
      end
      fract = gamma * 1/othercount2 # utilities are weighted by gamma and likelihood (for this problem, uniform over possibilities)
      a2r = R(theta, omega, 10) + fract * othercontrib2 # utility of this action

      U[i, j] = max(a1r, a2r)
    end
  end  
end



######### Construct policy #########

f = open("medium.policy", "w")
for (j, omega) in enumerate(omegas)
  for (i, theta) in enumerate(thetas)
      theta1, omega1 = T[i, j, 1]
      a1u = U[th2ind[theta1], om2ind[omega1]]
      theta2, omega2 = T[i, j, 2]
      a2u = U[th2ind[theta2], om2ind[omega2]]
      if a1u > a2u
        @printf(f, "0\n")
      else
        @printf(f, "10\n")
      end
  end
end
close(f)
