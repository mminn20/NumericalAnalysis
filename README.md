# GNN Project 

## 1) Message Passing Neural Network (MPNN)
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

Generalizing the convolution operator to irregular domains is typically expressed as a **neighborhood aggregation** or **message passing** scheme.

</br>

$x_i^{(k)} =  \gamma^{(k)} \space (x_i^{(k-1)}, \space \square_{j\in N(i)}\phi^{(k)} \space (x_i^{(k-1)}, \space x_j{(k-1)}, \space e_{j,i}))$

- $x_i^{(k-1)}$ : denotes node features of node $i$ in layer $(k-1)$ 
- $e_{j, i} \in R^D$ : edge features from node $j$ to node $i$ (optional) 
- $square$ : a differentiable, permutation invariant function, e.g. sum, mean, or max 
- $\gamma$ , $\phi$ : differentiable functions such as MLPs (Multiple Layer Perceptrons) 

</br>  

### MPNN variant

$x_i^{(k)}$ $=$ $\gamma^{(k)}$ $(CONCAT[x_i^{(k-1)},$ $\Sigma_{j\in\boxtimes(i)}$ $\phi^{(k)}(e_{j, i}\cdot(x_j^{(k-1)}-x_i^{(k-1)}))])$

</br>

***

## References 
- Creating Message Passing Networks
  
  https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

- Source code for `torch_geometric.nn.conv.message_passing`
  
  https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html#MessagePassing.aggregate

