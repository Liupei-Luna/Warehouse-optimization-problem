# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:09:03 2018

@author: Dell
"""

import time
import math
import numpy as np
import pandas as pd
from numpy import random


def OrderfromTable(OrderID, OrderFile):
    Order = OrderFile.loc[OrderFile[u'单据编号']==OrderID,:]
    return Order

def OrderInfo(Order):
    goodsTable = pd.DataFrame(Order[ u'商品代码'].values,columns = [u'商品代码'])
    goodsTable[u'颜色代码'] = Order[ u'颜色代码'].values
    goodsTable[u'尺码代码'] = Order[ u'尺码代码'].values
    goodsTable[u'数量'] = Order[ u'数量'].values
    address = Order[u'收货地址']
    assert goodsTable.index.is_unique, u'商品代码is not unique'
    assert len(set(address))==1, u'收货地址不一致'
    return goodsTable, address.values[0]
    
def stroeTable(goodsIDs,warehouse, storeFile,):
    storeNum = pd.DataFrame(np.zeros([len(warehouse), len(goodsIDs)]), columns = goodsIDs[u'商品代码'], index = warehouse)
    for ID in goodsIDs.values:
        
        stores = storeFile[[u'仓库代码',u'数量']][(storeFile[u'商品代码']==ID[0])&(storeFile[u'颜色代码']==ID[1])&(storeFile[u'尺码代码']==ID[2])]
        for house, num in stores[[u'仓库代码',u'数量']].values:
            storeNum.loc[house,ID[0]] += num
    
    return storeNum

def disGet(disatnceFile, warehouseDataframe, address):
    addsLL = disatnceFile.loc[disatnceFile['province'].str.contains(address[:2]),['longitude','latitude'] ]
    addsLL = list(addsLL.iloc[0])
    addsRR = []
    distance = []
    for i in range(30):
        add = disatnceFile.loc[disatnceFile['province'].str.contains(warehouseDataframe.iloc[i][u'仓库地址'][:2]),['longitude','latitude'] ]
        add = list(add.iloc[0])
        addsRR.append(add)
        dis = np.sqrt((addsLL[0]-addsRR[i][0])**2+(addsLL[1]-addsRR[i][1])**2)
        distance.append(dis)
    distance = pd.DataFrame(distance,index=warehouseDataframe.index,columns=['distance'])
    return distance

def costf(distance,x,priority):
    
    I30 = x.sum(axis=1)>0   

    w1 = 1.0
    w2 = 1.0
    w3 = 1.0

    f1 = (distance.values.T*I30.values).max()
    f3 = (priority.values.T*I30.values).sum()
    f = w1*f1 + w2*sum(I30)+ w3*f3
    return f


try:
    init += 1
except:
    init = 0
    print ('data loading...')
    
    ###########
    warehouseFile = pd.DataFrame(pd.read_excel(u'仓库档案.xlsx'))
    warehouse = warehouseFile[u'仓库代码']
    warehouseDataframe = warehouseFile.drop([u'仓库代码',], axis = 1,)
    warehouseDataframe = pd.DataFrame(warehouseDataframe.values, index = warehouse, columns = warehouseDataframe.columns)
    ###########
    storeFile = pd.DataFrame(pd.read_excel(u'库存数据.xlsx'))
    ###########
    OrderFile = pd.DataFrame(pd.read_excel(u'零售小票.xlsx'))
    OrderIDlist = OrderFile[u'单据编号']
    print ('loading complete')
    ###########
    with open(u'全国各区经纬度.csv',encoding='UTF-8') as f1:
        disatnceFile = pd.read_csv(f1)
        
#OrderID = 'xp20180808381863'
#Order = OrderfromTable(OrderID, OrderFile,)
#goodsTable, address = OrderInfo(Order)
#storeNum = stroeTable(goodsTable,warehouse, storeFile,)
#priority = warehouseDataframe.loc[warehouse,u'仓库优先级']
#distances = disGet(disatnceFile, warehouseDataframe, address)

def initially(goodsTable,storeNum):
    reserveNum = pd.DataFrame(storeNum.values.sum(axis=0)-goodsTable[u'数量'].values,
                              index=goodsTable[u'商品代码'])
    assert (reserveNum.values>=0).all(),u'库存不足'
    variableID = reserveNum.loc[reserveNum[0]>0].index
    
    X0 = []
    for i in range(10):   ##Np=10
        x = pd.DataFrame(np.zeros(storeNum.shape), index = storeNum.index,
                         columns = storeNum.columns)
        for goodsID in storeNum.columns:
            total = goodsTable[u'数量'][goodsTable[u'商品代码']==goodsID].values
            select_index = list(storeNum.loc[storeNum[goodsID]>0].index)
            random.shuffle(select_index)
            for storehouse in select_index:
                x.loc[storehouse,goodsID] = min(total,storeNum.loc[storehouse,goodsID] )
                total -= int(x.loc[storehouse,goodsID])
                if total == 0: break
        X0.append(x)
    if variableID.size == 0:
        print (u'只有唯一的发货方式！！！')
        
    return X0
        

def DE(Gm,F0,goodsTable,storeNum):
    
    Gm = 1  #最大迭代次数
    Np = 30
    F0 = 0.5  #变异率
    CR = 0.9 #交叉概率
    G = 1  #初始化代数
    D = len(goodsTable.index) #所求问题维数
    V=[]
    U=[]
    XG_next=[]
    value=np.zeros(Np)
    
    
    #产生初始种群 
    """
    reserveNum = pd.DataFrame(storeNum.values.sum(axis=0)-goodsTable[u'数量'].values,
                              index=goodsTable[u'商品代码'])
    assert (reserveNum.values>=0).all(),u'库存不足'
    variableID = reserveNum.loc[reserveNum[0]>0].index
    
    X0 = []
    x = pd.DataFrame(np.zeros(storeNum.shape), index = storeNum.index,
                         columns = storeNum.columns)
    for i in range(Np):
        X0.append(x)
    for goodsID in storeNum.columns:
    
        
        total = goodsTable[u'数量'][goodsTable[u'商品代码']==goodsID].values
        select_index = list(storeNum.loc[storeNum[goodsID]>0].index)
        t = 0
        for i in range(Np):
            
            sel = []
            num = 0
            
            for k in range(t,len(select_index)):
                sel.append(select_index[k])
                t +=1
                num += storeNum.loc[select_index[k],goodsID]
                if num >= total:
                    break
#            random.shuffle(select_index)
            for storehouse in sel:
                X0[i].loc[storehouse,goodsID] = min(total,storeNum.loc[storehouse,goodsID] )
                total -= int(X0[i].loc[storehouse,goodsID])
                if total == 0: break
#        X0.append(x)
    if variableID.size == 0:
        print (u'只有唯一的发货方式！！！')
    """    
        
    reserveNum = pd.DataFrame(storeNum.values.sum(axis=0)-goodsTable[u'数量'].values,
                              index=goodsTable[u'商品代码'])
    assert (reserveNum.values>=0).all(),u'库存不足'
    variableID = reserveNum.loc[reserveNum[0]>0].index
    
    X0 = []
    for i in range(Np):  
        x = pd.DataFrame(np.zeros(storeNum.shape), index = storeNum.index,
                         columns = storeNum.columns)
        for goodsID in storeNum.columns:
            total = goodsTable[u'数量'][goodsTable[u'商品代码']==goodsID].values
            select_index = list(storeNum.loc[storeNum[goodsID]>0].index)
            random.shuffle(select_index)
            for storehouse in select_index:
                x.loc[storehouse,goodsID] = min(total,storeNum.loc[storehouse,goodsID] )
                total -= int(x.loc[storehouse,goodsID])
                if total == 0: break
        X0.append(x)
    if variableID.size == 0:
        print (u'只有唯一的发货方式！！！')
    XG = X0  #初始种群
    
    while G <= Gm:
        print(G)
        #变异
        for i in range(1,Np):
            li = list(range(Np))
            random.shuffle(li)
            dx = li
            j = dx[0]
            k = dx[1]
            p = dx[2]
            if j==i:
                j = dx[3]
            elif k==i:
                k = dx[3]
            elif p==i:
                p = dx[3]
            #变异操作
            
            suanzi = math.exp(1-float(Gm)/(Gm+1-G))
            F = int(F0*(2**suanzi))
            mutant = XG[p]+F*(XG[j]-XG[k])
            for j in range(Np):
                if (np.array(mutant)).all()== (np.array(XG[j])).all(): 
                    V.append(mutant)
                else:
                    V.append(random.sample(XG))
                    
        #交叉操作
        for i in range(Np):
            randx = list(range(D))
            random.shuffle(randx)
            for j in range(D):
                if random.rand()>CR and randx[0] !=j:
                    U.append( XG[i])
                else:
                    U.append( V[i])
                    
        #选择操作
        for i in range(Np):
            f1 = costf(distances,XG[i],priority)
            f2 = costf(distances,U[i],priority)
            if f1<f2:
                XG_next.append(XG[i])
                value[i] = f1
            else:
                XG_next.append(U[i])
                value[i] = f2
        XG = XG_next
        G = G + 1        
            
            
    best_value = min(value)
    tmp = value.tolist()
    pos_min = tmp.index(min(tmp))
#    print(best_value)
    best_vector = XG[pos_min]
#    print(best_vector)
    return best_value,best_vector


error = []
sku = []  
x_all = [] 
OrderID_all = []
cost_min1 = []
#########################################################################
t1 = time.time()
for OrderID in OrderIDlist[:100].drop_duplicates():
    try:
        Order = OrderfromTable(OrderID, OrderFile,)
        goodsTable, address = OrderInfo(Order)  #订单信息，订单收货地址
        storeNum = stroeTable(goodsTable,warehouse, storeFile,) #对应商品仓库库存信息
        priority = warehouseDataframe.loc[warehouse,u'仓库优先级']
        distances = disGet(disatnceFile, warehouseDataframe, address)
        cost,x = DE(1,0.5,goodsTable,storeNum)
        x_all.append(x)
        OrderID_all.append(OrderID)
        cost_min1.append(cost)
        sku.append(list(goodsTable[[u'商品代码',u'颜色代码',u'尺码代码']].values[0]))
    except:
        error.append(OrderID)
t2 = time.time()
interval = t2-t1

#result_100 = pd.DataFrame(cost_min1,index=OrderID_all,columns=['min_cost'])
#    
#result_100.to_excel('result.xlsx')
#    
#
#for j in range(len(x_all)):
#    x_all[j] = x_all[j].values.T
#    
#result_100[u'商品代码'] = sku
#result_100[u'发货情况'] = x_all
#result_100.to_excel('result_100.xlsx')
            
    
    
    
    
    
    
    
    
    