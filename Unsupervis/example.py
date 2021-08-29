
from Unsupervis import *
dictAUC={}
dictPre={}

L='SmaGri'
print('data set...............',L,'for L=200')
G=nx.read_edgelist("Dataset\SmaGri.txt",create_using=nx.DiGraph(), nodetype = int)
data2,data3=Final_Result(G,L)
dictAUC.update({L:data2})
dictPre.update({L:data3})
DF=pd.DataFrame(dictAUC)
DF1=pd.DataFrame(dictPre)



L='Kohonen'
print('data set...............',L)
G=nx.read_edgelist("Dataset\Kohonen.txt",create_using=nx.DiGraph(), nodetype = int)
data2,data3=Final_Result(G,L)
dictAUC.update({L:data2})
dictPre.update({L:data3})
DF=pd.DataFrame(dictAUC)
DF1=pd.DataFrame(dictPre)

L='Political blogs'
print('data set...............',L,'For L=200..................')
G=nx.read_edgelist("Dataset\Political blogs.txt",create_using=nx.DiGraph(), nodetype = int)
data2,data3=Final_Result(G,L)

dictAUC.update({L:data2})
dictPre.update({L:data3})
DF=pd.DataFrame(dictAUC)
DF1=pd.DataFrame(dictPre)

L='Wiki-Vote'
print('data set...............',L,'For L=200..................')
G=nx.read_edgelist("Dataset\Wiki-Vote.txt",create_using=nx.DiGraph(), nodetype = int)
data2,data3=Final_Result(G,L)

dictAUC.update({L:data2})
dictPre.update({L:data3})
DF=pd.DataFrame(dictAUC)
DF1=pd.DataFrame(dictPre)



print('evaluation........................................')
strName1="result/Unsupervis_AUC.xlsx"
DF.to_excel(strName1,
           sheet_name='AUC')

strName1="result/Unsupervis_Precesion(L=200).xlsx"
DF1.to_excel(strName1,
           sheet_name='Pre')
