# molecule-cell-gen
Improve GDSS model by considering molecules as cellular complexes

Main tasks 

1. Consider adding more atomic features to $X_0$ before doing diffusion process

2. Lift the molecular graph $(X_0, A_0)$ to cellular complexes and update the time-dependent score models based on cell attention layers. For example, the cell attention network takes $X_0, X_1, A_0, A_{\uparrow, 1}$ and $A_{\downarrow, 1}$

