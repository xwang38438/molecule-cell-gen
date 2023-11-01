# molecule-cell-gen

## Installation:
- Follow the instructions in [TopoModelX](https://github.com/pyt-team/TopoModelX/tree/main)

## GDSS Overview:
A graph \(G\( with \(N\( nodes is defined by its node features \(X \in \mathbb{R}^{N \times F}\( and the weighted adjacency matrix \(A \in \mathbb{R}^{N \times N}\( as \(G = (X_0, A_0) \in \mathbb R^{N\times F} \times \mathbb R^{N \times N}:= \mathcal G\(, where \(F\( is the dimension of the node features. The diffusion process \(\left\{\boldsymbol{G}_t=\left(\boldsymbol{X}_t, \boldsymbol{A}_t\right)\right\}_{t \in[0, T]}\( transforms the node features and the adjacency matrices to a simple noise distribution. We can model the diffusion process by the following Ito SDE

\[
\mathrm{d} \boldsymbol{G}_t=\mathbf{f}_t\left(\boldsymbol{G}_t\right) \mathrm{d} t+\mathbf{g}_t\left(\boldsymbol{G}_t\right) \mathrm{d} \mathbf{w}, \quad \boldsymbol{G}_0 \sim p_{\text {data }},
\]
where \(\mathbf{f}_t(\cdot): \mathcal{G} \rightarrow \mathcal{G}^1\( is the linear drift coefficient, \(\mathbf{g}_t(\cdot): \mathcal{G} \rightarrow\( \(\mathcal{G} \times \mathcal{G}\( is the diffusion coefficient, and \(\mathbf{w}\( is the standard Wiener process. The reverse of the diffusion process in time is also a diffusion process described by the SDE:

\[
\mathrm{d} \boldsymbol{G}_t=\left[\mathbf{f}_t\left(\boldsymbol{G}_t\right)-g_t^2 \nabla_{\boldsymbol{G}_t} \log p_t\left(\boldsymbol{G}_t\right)\right] \mathrm{d} \bar{t}+g_t \mathrm{~d} \overline{\mathbf{w}}
\]
where \(p_t\( denotes the marginal  distribution under the forward diffusion process at time \(t\(, \(\bar{\mathbb w}\( is a reverse-time standard Wiener process, and \(\mathrm{d} \bar{t}\( is an infinitesimal negative time step. 

Solving __Eq. (2)__ requires the estimation of \(\nabla_{\boldsymbol{G}_t} \log p_t\left(\boldsymbol{G}_t\right) \in \mathbb{R}^{N \times F} \times \mathbb{R}^{N \times N}\( which is expensive to compute. So, the paper proposes a new reverse-time diffusion process.

\[
\left\{\begin{array}{l}
\mathrm{d} \boldsymbol{X}_t=\left[\mathbf{f}_{1, t}\left(\boldsymbol{X}_t\right)-g_{1, t}^2 \nabla_{\boldsymbol{X}_t} \log p_t\left(\boldsymbol{X}_t, \boldsymbol{A}_t\right)\right] \mathrm{d} \bar{t}+g_{1, t} \mathrm{~d} \overline{\mathbf{w}}_1 \\
\mathrm{~d} \boldsymbol{A}_t=\left[\mathbf{f}_{2, t}\left(\boldsymbol{A}_t\right)-g_{2, t}^2 \nabla_{\boldsymbol{A}_t} \log p_t\left(\boldsymbol{X}_t, \boldsymbol{A}_t\right)\right] \mathrm{d} \bar{t}+g_{2, t} \mathrm{~d} \overline{\mathbf{w}}_2
\end{array}\right.
\]
where \(\mathbf{f}_{1, t}\( and \(\mathbf{f}_{2, t}\( are linear drift coefficients satisfying \(\mathbf{f}_t(\boldsymbol{X}, \boldsymbol{A})=\left(\mathbf{f}_{1, t}(\boldsymbol{X}), \mathbf{f}_{2, t}(\boldsymbol{A})\right), g_{1, t}\( and \(g_{2, t}\( are scalar diffusion coefficients, and \(\overline{\mathbf{w}}_1, \overline{\mathbf{w}}_2\( are reverse-time standard Wiener Process. 

Note that the diffusion processes of \(X_0, A_0\( in the system are dependent on each other, related by the gradients of the joint log-density \(\nabla_{\boldsymbol{X}_t} \log p_t\left(\boldsymbol{X}_t, \boldsymbol{A}_t\right)\( and \(\nabla_{\boldsymbol{A}_{t}} \log p_t\left(\boldsymbol{X}_t, \boldsymbol{A}_t\right)\(, which are the __partial score functions__. The partial score functions can be estimated by training the time-dependent score-based models \(\boldsymbol{s}_{\theta, t}\( and \(\boldsymbol{s}_{\phi, t}\(, so that \(\boldsymbol{s}_{\theta, t}\left(\boldsymbol{G}_t\right) \approx \nabla_{\boldsymbol{X}_t} \log p_t\left(\boldsymbol{G}_t\right)\( and \(\boldsymbol{s}_{\phi, t}\(, so that \(\boldsymbol{s}_{\phi, t}\left(\boldsymbol{G}_t\right) \approx \nabla_{\boldsymbol{A}_t} \log p_t\left(\boldsymbol{G}_t\right)\(.

The paper proposes a novel training objective 
\[
\begin{aligned}
& \min _\theta \mathbb{E}_t\left\{\lambda_1(t) \mathbb{E}_{\boldsymbol{G}_0} \mathbb{E}_{\boldsymbol{G}_t \mid \boldsymbol{G}_0}\left\|\boldsymbol{s}_{\theta, t}\left(\boldsymbol{G}_t\right)-\nabla_{\boldsymbol{X}_t} \log p_{0 t}\left(\boldsymbol{X}_t \mid \boldsymbol{X}_0\right)\right\|_2^2\right\} \\
& \min _\phi \mathbb{E}_t\left\{\lambda_2(t) \mathbb{E}_{\boldsymbol{G}_0} \mathbb{E}_{\boldsymbol{G}_t \mid \boldsymbol{G}_0}\left\|\boldsymbol{s}_{\phi, t}\left(\boldsymbol{G}_t\right)-\nabla_{\boldsymbol{A}_t} \log p_{0 t}\left(\boldsymbol{A}_t \mid \boldsymbol{A}_0\right)\right\|_2^2\right\}
\end{aligned}
\]
wherewhere \(\lambda_1(t)\( and \(\lambda_2(t)\( are positive weighting functions and \(t\( is uniformly sampled from \([0, T]\(.  The expectations in Eq. (4) can be efficiently computed using the Monte Carlo estimate with the samples \((t, \boldsymbol G_0, \boldsymbol G_t)\(. 

Then, the paper proposes new architectures for the time-dependent score-based models that can capture the dependencies of \(\boldsymbol X_t\( and \(\boldsymbol A_t\( through time, based on \(\operatorname {GNNs}\(. The score-based model \(\boldsymbol s_{\phi,t}\( used to estimate \(\nabla_{\boldsymbol{A}_t} \log p_t\left(\boldsymbol{X}_t, \boldsymbol{A}_t\right)\( is defined by 

\[
\boldsymbol{s}_{\phi, t}\left(\boldsymbol{G}_t\right)=\operatorname{MLP}\left(\left[\left\{\operatorname{GMH}\left(\boldsymbol{H}_i, \boldsymbol{A}_t^p\right)\right\}_{i=0, p=1}^{K, P}\right]\right),
\]

where \(\boldsymbol{A}_t^p\( are the higher-order adjacency matrices, \(\boldsymbol{H}_{i+1}=\( \(\operatorname{GNN}\left(\boldsymbol{H}_i, \boldsymbol{A}_t\right)\( with \(\boldsymbol{H}_0=\boldsymbol{X}_t\( given, [.] denotes the concatenation operation, GMH denotes the graph multi-head attention block, and \(K\( denotes the number of \(\operatorname{GMH}\( layers. And the score-based model \(s_{\theta, t}\( to estimate \(\nabla_{\boldsymbol{X}_t} \log p_t\left(\boldsymbol{X}_t, \boldsymbol{A}_t\right)\( is defined as 
\[
\boldsymbol{s}_{\theta, t}\left(\boldsymbol{G}_t\right)=\operatorname{MLP}\left(\left[\left\{\boldsymbol{H}_i\right\}_{i=0}^L\right]\right),
\]
where \(\boldsymbol{H}_{i+1}=\operatorname{GNN}\left(\boldsymbol{H}_i, \boldsymbol{A}_t\right)\( with \(\boldsymbol{H}_0=\boldsymbol{X}_t\( given and \(L\( denotes the number of GNN layers. Since the message-passing operations of \(\operatorname{GNNs}\( and the attention function used in \(\operatorname{GMH}\( are permutation-equivariant, the score-based models are also equivariant, and the log-likelihood implicitly defined by the models is guaranteed to be permutation-invariant.


## Main tasks (order by priority)

__Intitution:__ Improve the GDSS model by considering molecules as cellular complexes, adding more atomic features and ring features. We want to know if topological deep learning can improve traditional graph representation learning. 

- Update code to lift molecular graph \(G:(X_0, A_0)\( to cellular complexes implemented by TopomodelX
    - Find the right place to conduct cellular lifting of the _GDSS_ code repo.
    -  Once we have a NetworkX graph, we can easily convert it to a CC object
    - Extract rings from the graph and add it to the CC object
        - Current Solution: use the graph-tool package to extract rings and add them to CC objects
        - Difficulty: the graph-tool package is not compatible with TMX now; this is a potential problem during training 

- Design two new time-dependent score-based models based on cell attention layers to estimate \(\nabla_{X_t}\log p_{0t}(G_t)\( and \(\nabla_{A_t}\log p_{0t}(G_t)\(
    - The cell attention network takes \(X_0, X_1, A_0, A_{\uparrow, 1}\( and \(A_{\downarrow, 1}\(, which can be extracted from the CC object converted from the molecular graph
    - The _CAN_ implemented in TopoModelX is designed for classification; we need to slightly update the model structure, including adding an MLP, to estimate a matrix with the same shapes \(X_0\( and \(A_0\(
    - Validate our new cell attention layer with Prof. Derr
    - Implement the new score functions in _ScoreNetwork_A.py_ and _ScoreNetwrok_X.py_


- Adding more atomic features to \(X_0\( before doing the diffusion process
    - Currently, the atom feature is just an array including the atomic numbers, e.g C:6 
    - Discuss with Lance to select the feature candidates derived from SMILEs
    - Embed the feature extraction method in _smile_to_graph.py_

- Since rank-2 cells (rings) are added, we should consider assigning attributes to them with the help of biological knowledge. (Optional)
    - With this new feature, we have to update the score function again. Thus, we should consider this as a later task


