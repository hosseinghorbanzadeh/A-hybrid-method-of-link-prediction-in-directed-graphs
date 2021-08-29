
from Method import *
from evaluation import Compear_Index
from darhamsaz import mix_row
L='SmaGri'


print('data set...............',L)

#G=nx.read_edgelist("Dataset\Political blogs.txt",create_using=nx.DiGraph(), nodetype = int)
G=nx.read_edgelist("Dataset\SmaGri.txt",create_using=nx.DiGraph(), nodetype = int)
CE=len(G.edges())-1
n=len(G.nodes())
complateEdges=n*(n-1)
NE=complateEdges-CE
Nnmber_Negative=(CE/(CE+NE))*NE
Number_Positive=(NE/(CE+NE))*CE

MissLink,WholeGraph_DF,TraningGraph_DF=Retuen_Miss_Link(G,0.3)
nonLink=Weight_sample(G,len(MissLink))

print('Weight_sample...','Number_Positive:',len(MissLink),'Nnmber_Negative Link',len(nonLink))
d=Final_Result_Multi_Dataset1(WholeGraph_DF,TraningGraph_DF,L,nonLink,MissLink)

data2=mix_row(d)
strName1="result/MLExcel"+" "+L+".xlsx"
data2.to_excel(strName1,
           sheet_name='MLExcel')

print('evaluation........................................')
data2= pd.read_excel(strName1, sheet_name='MLExcel')
Compear_Index(data2,L)


L='Political'


print('data set...............',L)

G=nx.read_edgelist("Dataset\Political blogs.txt",create_using=nx.DiGraph(), nodetype = int)

CE=len(G.edges())-1
n=len(G.nodes())
complateEdges=n*(n-1)
NE=complateEdges-CE
Nnmber_Negative=(CE/(CE+NE))*NE
Number_Positive=(NE/(CE+NE))*CE

print(CE)
print(Number_Positive)

MissLink,WholeGraph_DF,TraningGraph_DF=Retuen_Miss_Link(G,0.3)
nonLink=Weight_sample(G,len(MissLink))

print('Weight_sample...','Number_Positive:',len(MissLink),'Nnmber_Negative Link',len(nonLink))
d=Final_Result_Multi_Dataset1(WholeGraph_DF,TraningGraph_DF,L,nonLink,MissLink)

data2=mix_row(d)
strName1="result/MLExcel"+" "+L+".xlsx"
data2.to_excel(strName1,
           sheet_name='MLExcel')

print('evaluation........................................')
data2= pd.read_excel(strName1, sheet_name='MLExcel')
Compear_Index(data2,L)


L='wiki'


print('data set...............',L)

G=nx.read_edgelist("Dataset\Wiki-Vote.txt",create_using=nx.DiGraph(), nodetype = int)

CE=len(G.edges())-1
n=len(G.nodes())
complateEdges=n*(n-1)
NE=complateEdges-CE
Nnmber_Negative=(CE/(CE+NE))*NE
Number_Positive=(NE/(CE+NE))*CE


MissLink,WholeGraph_DF,TraningGraph_DF=Retuen_Miss_Link(G,0.3)
nonLink=Weight_sample(G,len(MissLink))

print('Weight_sample...','Number_Positive:',len(MissLink),'Nnmber_Negative Link',len(nonLink))
#def Final_Result_Multi_Dataset(WholeGraph,TraningGraph,L,non,miss):
d=Final_Result_Multi_Dataset1(WholeGraph_DF,TraningGraph_DF,L,nonLink,MissLink)

data2=mix_row(d)
strName1="result/MLExcel"+" "+L+".xlsx"
data2.to_excel(strName1,
           sheet_name='MLExcel')

print('evaluation........................................')
data2= pd.read_excel(strName1, sheet_name='MLExcel')
Compear_Index(data2,L)


L='Kohonen'


print('data set...............',L)

G=nx.read_edgelist("Dataset\Kohonen.txt",create_using=nx.DiGraph(), nodetype = int)

CE=len(G.edges())-1
n=len(G.nodes())
complateEdges=n*(n-1)
NE=complateEdges-CE
Nnmber_Negative=(CE/(CE+NE))*NE
Number_Positive=(NE/(CE+NE))*CE


MissLink,WholeGraph_DF,TraningGraph_DF=Retuen_Miss_Link(G,0.3)
nonLink=Weight_sample(G,len(MissLink))

print('Weight_sample...','Number_Positive:',len(MissLink),'Nnmber_Negative Link',len(nonLink))
#def Final_Result_Multi_Dataset(WholeGraph,TraningGraph,L,non,miss):
d=Final_Result_Multi_Dataset1(WholeGraph_DF,TraningGraph_DF,L,nonLink,MissLink)

data2=mix_row(d)
strName1="result/MLExcel"+" "+L+".xlsx"
data2.to_excel(strName1,
           sheet_name='MLExcel')

print('evaluation........................................')
data2= pd.read_excel(strName1, sheet_name='MLExcel')
Compear_Index(data2,L)
