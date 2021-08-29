import networkx as nx
import math
def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))

def adamic_adar_index(G, ebunch=None,TyppeGraph=None):
    if(TyppeGraph is None):
        print('NetworkX undirected graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            return sum(1 / math.log(G.degree(w))for w in nx.common_neighbors(G, u, v))
    elif(TyppeGraph=='DirGhraphIn'):
        print('NetworkX Admic-ader directed in graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            try:
                return sum(1 / math.log(G.in_degree(w)if G.in_degree(w)>0 else 1)for w in common_in_neighbors(G, u, v))
            except ZeroDivisionError:
                return 0
############################################################################
    elif(TyppeGraph=='DirGhraphOut'):
        print('NetworkX  Admic-ader directed out graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            try:
                return sum(1 / math.log(G.out_degree(w)if G.out_degree(w)>0 else 1)for w in common_out_neighbors(G, u, v))
            except ZeroDivisionError:
                return 0
    elif(TyppeGraph=='DirGhraphIn-Out'):
        print('NetworkX directed Admic-ader in and out graph')
        if ebunch is None:
            ebunch = nx.non_edges(G)
        def predict(u, v):
            try:
                sum1= sum(1 / math.log(G.out_degree(w)if G.out_degree(w)>0 else 1)for w in common_out_neighbors(G, u, v))
                sum2= sum(1 / math.log(G.in_degree(w)if G.in_degree(w)>0 else 1)for w in common_in_neighbors(G, u, v))
                sum3=sum1+sum2
                return sum3
            except ZeroDivisionError:
                 return 0
            
    else:
        print('Error in TyppeGraph')
         
    return ((u, v, predict(u, v)) for u, v in ebunch)
#WholeGraph_DF = pd.read_excel('lesmis.xlsx', sheet_name='out')
#TraningGraph_DF = pd.read_excel('lesmisTraning.xlsx', sheet_name='Sheet1')
'''g=[(1,2),(3,4),(1,3)]
G=nx.DiGraph()
G.add_edges_from(g)
print(list(adamic_adar_index(G,TyppeGraph='DirGhraphIn-Out')))'''
