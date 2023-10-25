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


## Model
### Supervised content augmentation of graph neural networks (AugS)
Figure presents the high level structure of our proposed model for content augmentation of graph neural networks, in the supervised setting. This
model is compatible with any graph neural network and does not depend
on the specific model used. As mentioned earlier, this model is primarily
effective in supervised scenarios, as deep auto-encoders tend to have limited
functionality in cases of data scarcity. We refer to this supervised content
augmentation model as AugS-GNN. In the following, we describe each component of the model in detail
![image](https://github.com/amkkashani/AugS-GNN/assets/32614364/a3f2cc79-85dc-4cec-9ddf-0273f05e49af)

### Structural and content embeddings
As mentioned earlier, we create two distinct embeddings for each node. First,
we utilize the GNN model, which incorporates both the graph structure and
features generated from the bag-of-words approach. In this way, a structural
embedding is built for each node. Then, in order to add a stronger dimension
of content information, at higher GNN layers and for each node, we generate
a content embedding. This is done by feeding the first-layer embedding (the
initial feature vector) of each node into an auto-encoder. The output of the
encoder component of the auto-encoder serves as the content embedding.

For auto-encoder, we use the model based on multiple MLP encoder layers
and multiple MLP decoder layers, introduced by Hinton [21]. In the encoder 
part, each layer’s size is reduced by half compared to the previous layer,
while the decoder part follows an opposite pattern, with each layer’s size
being doubled relative to the preceding one. The unsupervised loss function
used during training is defined by Equation 2, which helps train the the
model’s parameters:

![image](https://github.com/amkkashani/AugS-GNN/assets/32614364/2530ca61-6c21-4c7f-9067-210e6d58767c)


where Θ is the set of trainable parameters of the encoders and decoders,
operator ◦ is function composition, fenc and fdec respectively denote the
encoder and decoder functions (each consists of several MLP layers) and x
(i)
represents the initial feature vector of node i. The loss function computes the
squared L2 norm of the distance vector between the decoder’s composition of
the encoder’s output and the original input. Using it, we try to train the autoencoder parameters in a way that each vector, after encoding and decoding,
becomes close to its original form as much as possible. The summation
over n considers feature vectors of all nodes of the training dataset. After
completing the training phase, we use the output of the last encoder (the
input of the first decoder), as the content embedding. This layer is called
the bottleneck layer. It is worth highlighting that the parameters of the autoencoder are not jointly learned with the parameters of the whole model, as
a separate loss function is used to train the model.
We use the following setting to train the auto-encoder used to generate
content embeddings. We set the number of epochs to 1000, the input dimension based on number of features and the bottleneck dimension to 64. We
use Adam as the optimizer, ReLU as the activation function in the hidden
layers and softmax as the activation function in the output layer.

### Combination layer

In this layer, we combine the structural and content embeddings obtained
for each node, to form a unique embedding for it. Our combination layer
consists of two phases: the fusion phase, wherein the two structural and
content embeddings are fused to form a single vector; and the dimension
reduction phase, wherein the dimentionality of the vector obtained from the
first phase is reduced. For the first phase, we can explore several fusion
functions, including: concatenation where the two vectors are concatenated,sum where an element-wise sum is applied to the vectors and max where an
element-wise max is applied to the vectors. Our Experiments demonstrate
that concatenation consistently outperforms the other fusion methods across
all datasets. Therefore in this paper, we specifically highlight the results
achieved using the concatenation function.
For the second phase, we incorporate an MLP, whose parameters are
trained jointly with the other parameters of the model (unlike the parameters
of the auto-encoder used to generate the content embedding).


### Prediction head

The embeddings generated by augmented GNN models can be used as the
input for several downstream tasks and problems such as classification, regression, and sequence labeling. The method used to address these tasks
is called the prediction head. Various machine learning techniques, such as
SVD, decision trees, and linear regression, can be used in the prediction
process. In the experiments of this paper, our focus is to assess the model’s
performance in classifying nodes. To do so, we use an MLP, consisting of two
dense hidden layers each one with 16 units, as the prediction head. The parameters of the prediction head are jointly trained with the other parameters
of the whole model.
We train our model using the cross entropy loss function [22]. For a single
training example, it is defined as follows:
![image](https://github.com/amkkashani/AugS-GNN/assets/32614364/e3a9ac97-dbd5-43aa-a5e6-e630f8a4c83b)

where c is the number of classes and Tk and Sk respectively represent the
true probability and the estimated probability of belonging the example to
class k. The total cross entropy is defined as the sum of cross entropies of
all training examples.
We use the following setting to train the model. We set the number of
epochs to 200, batch size to 32, dropout ratio to 0.05 (in the MLP of the
prediction head, we set it to 0.2), and the number of GNN layers to 2. We use
the Adam algorithm as the optimizer and ReLU as the activation function
in the hidden layers and softmax in the output layer.



