# A-hybrid-method-of-link-prediction-in-directed-graphs
Link prediction is an important issue in complex network analysis and mining. Given the structure of a network, a
link prediction algorithm obtains the probability that a link is established between two non-adjacent nodes in the
future snapshots of the network. Many of the available link prediction methods are based on common neighborhood.
A problem with these methods is that if two nodes do not have any common neighbors, they always
predict a chance of zero for establishment of a link between them; however, such nodes have been shown to
establish links in some real systems. Another issue with these measures is that they often disregard the
connection direction. Here, we propose a novel measure based on common neighborhood that resolves the above
issues.
# Proposed measure
The method proposed in this paper introduces a novel measure by
combining the features obtained based on common neighbors and the
hubs and authorities of the nodes. The proposed method overcomes the
limitations of common neighbor based measures, while maintaining
their simplicity. In the proposed measure, the hub, authority, and direction
of the connection (in-out-neighbor) are used along with the information
on the common neighbors
 # CN-AH_AH
 is a Python library which offers for Proposed measure. 
The measures proposed in this paper are compared to the following baseline methods:CN-IN,CN-OUT,AA-IN,...
The performance of the proposed measure is evaluated in two modes: Unsupervised link prediction and Supervised link prediction
A Article of  Proposed measure can be found in https://www.sciencedirect.com/science/article/abs/pii/S0957417420306965
# Graph Format
The format of the graph is text.for example:
1 2
1 3
...
# Required libraries



