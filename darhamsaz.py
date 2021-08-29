import pandas as pd 

# Define a dictionary containing employee data 

import random
def RandomNumber(Low,Hight,Num):
    L=[]
    i=0
    while i<Num:
        r=random.randint(Low,Hight)
        if(r not in L):
            L.append(r)
        i+=1
    return L

def random_Bedoune_Tekrar(N,nb_sample):

    """ random dar majmoueh {0,1,...N}
    bedoune Tekrar.  nb_nsample < N  """
    import random
    import math
    data = []
    n = 1
    while 1:
        U=math.floor(N*random.random())
        if U not in data: data.append(U)
        else : continue
        n += 1
        if n > nb_sample: break
    return data

#MLExcel SmaGri
#def mix_row(str_path,Sheet_Name):
def mix_row(data1):
    #data1= pd.read_excel('MLExcel Political.xlsx', sheet_name='MLExcel')
    #data1= pd.read_excel(str_path, sheet_name=Sheet_Name)
    #print(str_path,'.............................')
    print('=========================')
    lenData=len(data1)-2
    print(lenData)
    LD=int(lenData/2)
    RN=random_Bedoune_Tekrar(lenData,lenData-20)
    print('RN',len(RN))
    print('data',len(data1))
    rn=len(RN)-2
    for i in range(rn):
        k1= RN[i]
        k2= RN[i+1]
        temp=data1.iloc[k1].copy()
        data1.iloc[k1]=data1.iloc[k2]
        data1.iloc[k2]=temp
    #data1.to_excel(str_path, sheet_name=Sheet_Name)
    return data1
######################################################################
#data1=(mix_row('MLExcel Political.xlsx','MLExcel'))
#data1= pd.read_excel('MLExcel Political.xlsx', sheet_name='MLExcel')
#print(data1)