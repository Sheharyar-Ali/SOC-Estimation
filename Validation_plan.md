## Overview

The best way to validate this model is by comparing the results to the car's shunt's coulomb counter.
This way we can see the difference in the energy use stated by the model and the energy use shown by the Coulomb counter.

## Experimental validation

This procedure involves charging and discharging the battery, in small intervals, and using the current and voltage readings to see how much energy ahs been expended.

First, the battery must be charged to full to get a calibration point for the SOC.
Then it must be drained for 2 minutes and allowed t rest for 5 minutes. This ensures that the OCV is stable during our readings. 
The voltage and current (and current counter) readings from the shunt must then be recorded.
This needs to continue until the battery is completely drained

This must then be done for the charging of the accumulator to further improve the validation.

The data can be used in the SOC-Estimation to see how much energy has been used up and then use the start and end points of the experiment as the start and end points of the SOC.
The data from the Coulomb counter can then used to check how much energy was used and we can compare this value to the estimate from the SOC Estimation.

