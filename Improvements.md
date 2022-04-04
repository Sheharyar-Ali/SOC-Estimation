# Potential Improvements
This document will detail the potential improvements that can me made to the model by discussing some of the issues that arose

## Problem 1: Hysterisis
Currently the model assumes that the charge and discharge profile is the same and thus we ignore hysterisis.
This is obviously not accurate and thus we need to be able to model it. This can be done easily by charging 
and discharging the battery (See Problem 3)

## Problem 2: Q matrix 
Currently, the Process Noise matrix, Q has been estimated based off literature which leads to some inaccuracies.
This matrix should ideally be calculated based on the standard deviations of all the parameters used in the state equations.

These parameters include: `[R0, R1, R2, C1, C2, Cap]`. Once we have the standard deviations of these parameters, we can find out the
process noise matrix by using the follow ing equation:

where `Qp` is the matrix containing the standard deviations and `J` is given by:



## Problem 3: Relationship between OCV and SOC

## Problem 4: Create more Lookup Tables