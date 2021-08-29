import random
from itertools import islice
import operator
def Measure_Precision(SmissLink1,Snonexits1,L):
    Smarge=SmissLink1+Snonexits1
    Sort_Smarge=sorted(Smarge,key=operator.itemgetter(2),reverse = True)
    Pick_Smarge=list(islice(Sort_Smarge, L))
    #print(Pick_Smarge)
    print('$$$$$$$$$$$$$$$$$$$$$')
    k=0
    for i in range(len(SmissLink1)):
        for j in range(len(Pick_Smarge)):
            if SmissLink1[i][0]==Pick_Smarge[j][0]:
                if SmissLink1[i][1]==Pick_Smarge[j][1]:
                    #print(SmissLink1[i])
                    k=k+1
                    break
    return(round((k/L),4))
