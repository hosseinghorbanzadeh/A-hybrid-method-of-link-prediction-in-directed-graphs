import networkx as nx
import math
def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))

def resource_allocation_index(G, ebunch=None,TyppeGraph=None):
    if(TyppeGraph is None):
        print('NetworkX undirected graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            return sum(1 /(G.degree(w))for w in nx.common_neighbors(G, u, v))
    elif(TyppeGraph=='DirGhraphIn'):
        print('resource_allocation directed in graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            try:
                return sum(1 /(G.in_degree(w))for w in common_in_neighbors(G, u, v))
            except ZeroDivisionError:
                #print('Zero')
                return 0
            
############################################################################
    elif(TyppeGraph=='DirGhraphOut'):
        print('resource_allocation directed out graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            try:
                return sum(1 /(G.out_degree(w))for w in common_out_neighbors(G, u, v))
            except ZeroDivisionError:
                return 0
            
    elif(TyppeGraph=='DirGhraphIn-Out'):
        print('resource_allocation directed in out graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            try:
                sum1 = sum(1 /(G.out_degree(w))for w in common_out_neighbors(G, u, v))
                sum2 = sum(1 /(G.in_degree(w))for w in common_in_neighbors(G, u, v))
                sum3=sum1+sum2
                return sum3
            except ZeroDivisionError:
                return 0
                
    else:
        print('Error in TyppeGraph')
         
    return ((u, v, predict(u, v)) for u, v in ebunch)
#WholeGraph_DF = pd.read_excel('lesmis.xlsx', sheet_name='out')
#TraningGraph_DF = pd.read_excel('lesmisTraning.xlsx', sheet_name='Sheet1')
#g=[(1,2),(3,4),(1,3)]
'''g=[(1,3),(2,3),(4,1),(4,2)]
G=nx.DiGraph()
G.add_edges_from(g)
print(list(resource_allocation_index(G,TyppeGraph='DirGhraphIn-Out')))'''
