#import "../extra.typ": definition


== Animal movement models

== Day-Night Cycle Model

From this point on, we will introduce a day-night cycle model that captures the essential features of the day-night cycle and then use it to explore how this cycle can influence the population dynamics.

#definition(title: [Day-Night Cycle Model])[
  Let a period $T>0$ representing the duration of a cycle of 24 hours. We define a partition of the interval $I = [0, T]$ into two subintervals $I_"day"$ and $I_"night"$ corresponding respectively to a period of daylight and a period of darkness.
] <definition-day-night-cycle-model>

It's clear that these two intervals are not necessarily of the same length, and that they can vary depending on the time of the year and the geographical location. For simplicity, we will assume that they are fixed and that $I_"day" = [0, t_"day"]$ and $I_"night" = [t_"day", T]$ for some $t_"day" in (0, T)$.

We can also consider the fact that for this model there is a sharp transition between day and night, but in reality, there is a gradual transition between the two states. However, for the sake of simplicity, we will stick to the sharp transition model.

== Perception

== Behavior of species

With the day and night cycle, we can have two different species that can appear: the nocturnal and diurnal species. 

#definition(title: [Nocturnal - Biology Based])[
  A population is called nocturnal if it is active during the night and inactive during the day.
] <definition-nocturnal-biology-based>

#definition(title: [Diurnal - Biology Based])[
  A population is called diurnal if it is active during the day and inactive during the night.
] <definition-diurnal-biology-based>

We consider that a population is inactive if and only it does not move. That means that the diffusion coefficient is zero during the inactive period and positive during the active period. We have the same behavior for the advection coefficient.