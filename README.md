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
 # CN_AH_AH
 is a Python code which offers for Proposed measure. 
The measures proposed in this paper are compared to the following baseline methods:CN-IN,CN-OUT,AA-IN,...
The performance of the proposed measure is evaluated in two modes: Unsupervised link prediction and Supervised link prediction
A Article of  Proposed measure can be found in https://www.sciencedirect.com/science/article/abs/pii/S0957417420306965
# Graph Format
The format of the graph is text.for example:                                                                                                                                        
1 2                                                                                                                                                                                  
1 3                                                                                                                                                                                  
...
# Required libraries
CN_AH_AH is tested to work on Python 3.6.
Libraries to be installed:
pandas,
numpy,
operator,
networkx,
matplotlib,
xlsxwriter,
itertools,
sklearn, 
scipy,
random,
math,
ast,
node2vec.
Or you can use the command:
pip install requirements.txt
# Run Code
package CN_AH_AH.zip contains two folders:                                                                                                                                          supervis                                                                                                                                                                            unsupervis                                                                                                                                                                          In each folder you can see the evaluation in folder result by running the example file.
# Cite
@article{                                                                                                                                                                           
    title = "A-hybrid-method-of-link-prediction-in-directed-graphs",                                                                                                                
    journal = "Expert Systems With Applications",                                                                                                                                    
    year = "2021",                                                                                                                                                                  
    issn = "0957-4174",                                                                                                                                                              
    doi = "https://doi.org/10.1016/j.eswa.2020.113896",                                                                                                                              
    url = "https://www.sciencedirect.com/science/article/abs/pii/S0957417420306965",                                                                                                 
    author = "Hossein ghorbanzade ,Amir Sheikhahmadi,Mahdi Jalili and Sadegh Sulaimanyc",                                                                                            
    keywords = "Link Prediction ,Structural Similarity,Local Similarity Measure,Common Neighborhood,Supervised Learning,Unsupervised Learning"                                       
}
