import networkx as nx
from node2vec import Node2Vec
def n2vlp(G_data,Edges):
  sim_Edges=[]
  node2vec = Node2Vec(G_data, dimensions=100, walk_length=100, num_walks=18)
  n2w_model = node2vec.fit(window=7, min_count=1)
  for e in Edges:
    sim_Edges.append((e[0],e[1],n2w_model.wv.similarity(str(e[0]),str(e[1]))))
  return(sim_Edges)

