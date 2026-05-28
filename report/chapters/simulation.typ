#import "@preview/diverential:0.3.0": *

== Model for one population

Putting all our thoughts so far together, if we consider only one population, we can write down the following model:

$
  cases(
  dvp(u,t) (x,t) = d(t) laplace u(x,t) - chi(t) nabla (u(x,t) nabla h(x,t)),

  h(x,t) = (w K_"sight" (x,t) + (1-w) K_"smell" (x,t)) * u(x,t)
  )
$

For the fist simulations, we will consider a case where the population is moving in one dimension. 