# AugS-GNN

## abstract

In recent years, graph neural networks (GNN) have become a popular tool for solving various problems over
graphs. The link structure of the graph is typically exploited by existing GNNs, and the embeddings of nodes
are iteratively updated based on their embeddings and the embeddings of their neighbors in the previous
iteration (layer). Nodes’ contents are used solely in the form of feature vectors, which serve as the nodes’
first-layer embeddings. However, the consecutive filters (convolutions) applied during iterations/layers to
these initial embeddings lead to their impact diminishing and contribute insignificant to the final embeddings.
To address this issue, in this paper, we propose augmenting node embeddings with embeddings generated
from their content in the final phase. More precisely, an embedding is extracted from the content of each node
and concatenated with the embedding generated from the graph neural network. An unsupervised dimension
reduction method based on auto-encoders is employed to reduce the dimensions of the generated embeddings
to a desired value. Our method is independent of the specific GNN used and can be aligned with any of them.
In the end, through experiments conducted on several real-world datasets, it is demonstrated that our method
significantly improves the accuracy and performance of GNNs.
