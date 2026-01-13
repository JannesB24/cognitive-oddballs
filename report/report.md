# Report 

## Environments

We created two oddball environments for a predictive inference task. The task involves the participants or agents to infer the position of a helicopter based on the position of a bag that falls periodically from it. What makes the task interesting is that the helicopter is obscured behind clouds.
Two types of noise apply to this task: (1) observation noise, which is the noise in the position of the bags relative to the helicopter, and (2) environmental noise, which is the noise in the movement of the helicopter itself. The latter noise is produced by sudden changes in the position of the helicopter (change points) or gradual drifts in its position (Gaussian random walk), while the former noise is produced by sampling the bag positions from a Gaussian distribution centered on the helicopter position.

In addition, at random intervals an oddball bag will fall, which is sampled from a uniform distribution over the entire screen width. This bag drop is not associated with the helicopter position and thus provides no information about it.

### Common environmental features

The width of the environment is sampled at 501 ($x \in [0, 500]$) positions, which are allowed bag positions. The helicopter is located is more restricted between $3 * \sigma$ and $500 - 3 * \sigma$ to avoid edge effects, where $\sigma$ is the standard deviation of the observation noise. 

### Change-point environment




### Gaussian random walk environment