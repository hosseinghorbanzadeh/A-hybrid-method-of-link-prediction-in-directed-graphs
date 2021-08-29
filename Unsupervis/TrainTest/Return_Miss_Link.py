import networkx as nx
#from Sampling_nonedges import *
from TrainTest.Sampling_nonedges  import iterSample
def Retuen_Miss_Link(WholeGraph,alpha):
    #WholeGraph=nx.DiGraph()
    #TraningGraph=nx.DiGraph()
    #WholeGraph.add_edges_from(WholeGraph_df)
    #TraningGraph.add_edges_from(WholeGraph_df)
    TraningGraph=WholeGraph.copy()
    edges=WholeGraph.edges()
    #print(edges)
    Number=int(len(edges)*alpha)
    sample=iterSample(edges,Number)
    TraningGraph.remove_edges_from(sample)
    #l=TraningGraph.edges()
    #print('TraningGraph',TraningGraph.edges())
    set1=set(WholeGraph.nodes())
    set2=set(TraningGraph.nodes())
    set3=set1.difference(set2)
    list1=list(set3)
    if list1!=[]:
        for i in range(len(list1)):
            TraningGraph.add_node(list1[i])
    G2=nx.difference(WholeGraph,TraningGraph)
    missLink=list(G2.edges())
    return missLink,WholeGraph,TraningGraph
#Example
#WholeGraph_DF=[(1,3),(1,5),(2,3),(2,4),(2,5),(3,5),(5,4)]
#WholeGraph=nx.DiGraph()
#WholeGraph.add_edges_from(WholeGraph_DF)
#missLink,WholeGraph,TraningGraph=Retuen_Miss_Link(WholeGraph,0.5)
#print('missLink',missLink)
