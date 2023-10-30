# molecule-cell-gen

__GDSS Overview__:
A graph $G$ with $N$ nodes is defined by its node features $X \in \mathbb{R}^{N \times F}$ and the weighted adjacency matrix $A \in \mathbb{R}^{N \times N}$ as $G = (X,A) \in \mathbb{N\times F} \times \mathbb R^{N \times N}:= \mathcal g$


Improve GDSS model by considering molecules as cellular complexes

__Idea:__ We utilize the original framework of _GDSS_, in which $X_0$ and $A_0$ incorporated with edge features are generated for new molecules. _GDSS_ constructs a system of SDEs for these two components to gradually adding noise to the molecular graph. To capture the dependency between $X_0$ and $A_0$ through time, _GDSS_ estimate the 

score function $s_{\theta,t}(G_t)$

feeds them into $\operatorName{GNNs}$ 


__Main tasks:__

- Represent molecules with cellular complexes implemented by TopomodelX. Since rank-2 cells (rings) are added, we should consider assigning attributes to them with the help of biological knowledge. 

- Consider adding more atomic features to $X_0$ before doing diffusion process.

- Lift the molecular graph $(X_0, A_0)$ to cellular complexes and update the time-dependent score-based models based on cell attention layers. For example, the cell attention network takes $X_0, X_1, A_0, A_{\uparrow, 1}$ and $A_{\downarrow, 1}$, which can be extracted from the molecular graph. 
