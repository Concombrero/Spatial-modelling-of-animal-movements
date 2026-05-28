#import "../extra.typ": definition, lemma, proof
#import "@preview/diverential:0.3.0": *

== Animal movement model
Before introducing the day-night cycle model, we will first introduce some basic animal movement models that will be used as a basis @wang2023openproblemspdemodels. We note $u(x,t)$ the population density at position $x$ and time $t$. We then consider the following general model for the population dynamics of $n$ species:

$
  dvp(u, t) (x,t) = d Delta u(x,t) - nabla dot (u(x,t) nabla a(x,t))
$ <equation-animal-movement-model>

The diffusion term $d$ represents the random movement of the individuals, while the advective potential $a(x,t)$ corresponds to the bias in movement based on information. In our case, we will consider that the advective potential will only depend on the perception of the environment.

=== Perception Model

The goal of the model is to capture the ability of the individuals to gather informations about their environment via non-local perception and use it to bias their movement. For example, an individual can perceive the presence of resources, predators or conspecifics and use this information to move towards or away from them. The "thing" that can be perceived is represented by a density function $m(x,t)$. The perception of the environment is then captured by a kernel $K$ that represents the perceptual field of the individuals. We can then describe the perception of the environment by the following equation:

$
  h(x,t) := K(x,t) * m(x,t) = integral_Omega K(x-y,t) m(y,t) d y
$<equation-perception-model>

$h$ can be interpreted as the perceived density of the "thing" at position $x$ and time $t$. The advective potential can then be defined as a function of the perceived density $h$, for example $a(x,t) = chi h(x,t)$. This means that the individuals will move towards areas where the perceived density of the "thing" is higher if $chi > 0$ and away from them if $chi < 0$. $chi$ mesure the attraction or repulsion of the populationh to the "thing" that is perceived. Then the model can be written as follows:

$
  dvp(u, t) (x,t) = d Delta u(x,t) - chi nabla dot (u(x,t)  nabla h(x,t))
$

== Day-Night Cycle Model

From this point on, we will introduce a day-night cycle model that captures the essential features of the day-night cycle and then use it to explore how this cycle can influence the population dynamics.

#definition(title: [Day-Night Cycle Model])[
  Let a period $T>0$ representing the duration of a cycle of 24 hours. We define a partition of the interval $I = [0, T]$ into two subintervals $I_"day"$ and $I_"night"$ corresponding respectively to a period of daylight and a period of darkness.
] <definition-day-night-cycle-model>

It's clear that these two intervals are not necessarily of the same length, and that they can vary depending on the time of the year and the geographical location. For simplicity, we will assume that they are fixed and that $I_"day" = [0, t_"day"]$ and $I_"night" = [t_"day", T]$ for some $t_"day" in (0, T)$.

We can also consider the fact that for this model there is a sharp transition between day and night, but in reality, there is a gradual transition between the two states. However, for the sake of simplicity, we will stick to the sharp transition model.

#figure(image("../figures/day_night_cycle_test.png", width: 30%), caption: "Illustration of the Day-Night Cycle Model")

Perception depends on the environment then we assume that the day-night cycle can affect the perception. Thus when we talk about a change of perception, we are talking about a change of the perceptual kernel $K$ that captures the environment. Then the cycle can be captured by a kernel of perception that depends on time and that changes between the day and night intervals.

== Perception

[INTRO PERCEPTION]

=== Combination of perception
In reality, animals can have multiple types of perception that can be affected or not by the day-night cycle. We can then consider a combination of different types of perception to capture the overall perception of the environment.

#lemma()[
  Let ${K_i}_i$ be a family of kernels of perception and let ${w_i}_i$ be a family of weights such that $w_i > 0$ for all $i$ and $sum_i w_i = 1$. We have that the combined kernel of perception defined as follows:

  $
    K(x-y, t) = sum_i w_i K_i(x-y, t)
  $

  is a kernel of perception.
]<lemma-combination-of-perception-kernels>

#proof()[
  We reacall that a perceptual kernel satisfies the following properties:

  + $K$ is symmetric in the spatial variable
  + $integral_Omega K(x, t) d x = 1$ for all $t$
  + $lim_(R->0^+) K(x, t) = delta(x)$ for all $t$
  + $K$ is non increasing from the origin in the spatial variable

  We will show that the combined kernel of perception defined above satisfies these properties @wang2023openproblemspdemodels.

  + A sum of symmetric functions is symmetric, so $K$ is symmetric in the spatial variable.

  + By linearity, we have
    #math.equation(block: true, numbering: none,
      $
        integral_Omega K(x, t) d x
          &= integral_Omega sum_i w_i K_i(x, t) d x \
          &= sum_i w_i integral_Omega K_i(x, t) d x \
          &= sum_i w_i \
          &= 1
      $
    )
    for all $t$.

  + By linearity of limits, we have
    #math.equation(block: true, numbering: none,
      $
        lim_(R->0^+) K(x, t)
          &= lim_(R->0^+) sum_i w_i K_i(x, t) \
          &= sum_i w_i lim_(R->0^+) K_i(x, t) \
          &= sum_i w_i delta(x) \
          &= delta(x)
      $
    )
    for all $t$.

  + A sum of non increasing functions is non increasing, so $K$ is non increasing from the origin in the spatial variable.
]

We can then use this lemma to combine different types of perception to capture the overall perception of the environment. In this article, we will focus only on the sight-based and smell-based perception.

$
  K(x-y,t) = w K_"sight" (x-y,t) + (1-w) K_"smell" (x-y)
$

This simple combination allows us to capture the effects of the day-night cycle on the perception of the environment and compare it to a situation where it's not affected by the change of light.


=== Sight-based perception
It's clear that the perception of the environment can be strongly affected by the day-night cycle. For example, a animal that relies on vision to perceive it's environment will have a much better perception during the day than during the night. Based on this observation we can assume that during the night the individuals cannot perceive anything, which means that the sight-based perception kernel is zero during the night.It's clear that this type of kernel is greatly simplified, but it captures the essential features of the perception during the day and night. That leaves one question: how does the kernel behave during the day? 

$
  K_"sight" (x-y, t) = cases(
    K(x-y) &quad "if" t in I_"day",
    0 &quad "if" t in I_"night"
  )
$<equation-sight-based-perception-kernel>

=== Smell-based perception
On the other hand, we assume that the perception of the environment based on smell is not affected by the day-night cycle, which means that the smell-based perception kernel is the same during the day and night. Thus it will be only a function of the spatial variable and not of time. When it comes to making a choice, the spread of odors follows highly diffusive dynamics and thus a Gaussian kernel is a good approximation of the smell-based perception kernel @Baker9383.

$
  K_"smell" (x) = frac(1, sqrt(2 pi) R) e^(x^2 slash 2 R^2)
$<equation-smell-based-perception-kernel>

== Behavior of species

With the day and night cycle, we can have two different species that can appear: the nocturnal and diurnal species. 

#definition(title: [Inactivity of a population])[
  We consider that a population is inactive during a period of time $[t_1, t_2]$ if and only if the population density is constant during this period of time, i.e. $u(x,t) = u(x,t_1)$ for all $t$ in $[t_1, t_2]$ and for all $x$ in the spatial domain.
]<definition-inactivity-of-a-population>

#definition(title: [Nocturnal - Biology Based])[
  A population is called nocturnal if it is active during the night and inactive during the day.
] <definition-nocturnal-biology-based>

#definition(title: [Diurnal - Biology Based])[
  A population is called diurnal if it is active during the day and inactive during the night.
] <definition-diurnal-biology-based>


Thus to model the behavior of a inactive population, we can fix the diffusion coefficient and the mesure of attraction to zero during the period of inactivity. For example, for a nocturnal species, we can assume that:

#math.equation(block: true,
  grid(
    columns: 2,
    gutter: 2em,
    [$D(t) = cases(
      D_"active" &quad "if" t in I_"night",
      0 &quad "if" t in I_"day"
    )$],
    [$chi(x,t) = cases(
      chi_"active" (x) &quad "if" t in I_"night",
      0 &quad "if" t in I_"day"
    )$],
  )
)