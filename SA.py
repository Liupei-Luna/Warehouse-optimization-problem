# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:19:49 2018

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:06:01 2018

@author: vlev0
"""
import pandas as pd 
import numpy as np
from collections import Counter
import math
from random import seed
import random
import time
from scipy.sparse import csr_matrix
random.seed(0)

def OrderfromTable(OrderID, OrderFile):
    Order = OrderFile.loc[OrderFile[u'单据编号']==OrderID,:]
    return Order

def OrderInfo(Order):
    goodsTable = pd.DataFrame(Order[ u'数量'].values,index = Order[u'商品代码'],)
    address = Order[u'收货地址']
    assert goodsTable.index.is_unique, u'商品代码is not unique'
    assert len(set(address))==1, u'收货地址不一致'
    return goodsTable, address.values[0]
    
def stroeTable(goodsIDs,warehouse, storeFile,):
    storeNum = pd.DataFrame(np.zeros([len(warehouse), len(goodsIDs)]), columns = goodsIDs, index = warehouse)
    for ID in goodsIDs:
        stores = storeFile.loc[storeFile[u'商品代码']==ID,:]
        for house, num in stores[[u'仓库代码',u'数量']].values:
            storeNum.loc[house,ID] += num
    
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

def randChange(x, storeNum, variableID):
    assert variableID.size >0, u'没有SKU的发货方式可供调整'
    goodsID = random.sample(list(variableID), 1)[0]
    from_Num = x.loc[x[goodsID]>0, goodsID,]
    from_house = random.sample(list(from_Num.index), 1)[0]
    to_Num = storeNum[goodsID]-x[goodsID]
    to_Num = to_Num[to_Num>0]
    if len(to_Num)==0:
        print ('len(to_Num)==0')
        return x
    else:
        to_house = random.sample(list(to_Num.index), 1)[0]
        
        num = random.randint(1,min(from_Num[from_house], to_Num[to_house]))
        
        x.loc[from_house,goodsID] -= num
        x.loc[to_house,goodsID]   += num
        
        
        
        assert ((storeNum.values-x.values) >= 0 ).all()
        return x

#初始化x
def initially(goodsTable,storeNum):
    reserveNum = pd.DataFrame(storeNum.values.sum(axis=0)-goodsTable.values[:,0], index = goodsTable.index)
    assert (reserveNum.values>=0).all(), u'库存不足'
    variableID = reserveNum.loc[reserveNum[0]>0].index
    x = pd.DataFrame(np.zeros(storeNum.shape), index = storeNum.index, columns = storeNum.columns)
    for goodsID in storeNum.columns:
        total = goodsTable.loc[goodsID, 0]
        for storehouse in storeNum.index:
            x.loc[storehouse,goodsID] = min(total,storeNum.loc[storehouse,goodsID] )
            total -= x.loc[storehouse,goodsID]
            if total == 0: break
    if variableID.size == 0:
        print (u'只有唯一的发货方式！！！')
    return x, variableID


def costf(distance,x,priority):
    
    I30 = x.sum(axis=1)>0

    w1 = 1.0
    w2 = 1.0
    w3 = 1.0

    f1 = (distance.values.T*I30.values).max()
    f3 = (priority.values.T*I30.values).sum()
    f = w1*f1 + w2*sum(I30)+ w3*f3
    return f


def annealingoptimize(x,storeNum,distance,priority,variableID,T=1000.0,cool=0.95,iterMax=10000):
    x_min = x.copy()
    
    cost0 = costf(distance,x,priority)
    cost_min = cost0
    
    
    
    while T>0.1:
        for i in range(iterMax):
            
            x_new = randChange(x.copy(),storeNum,variableID)
#            cost0 = costf(distance,x,priority)
        
            # Calculate the current cost and the new cost
            cost1 = costf(distance,x_new,priority)
            
            p=np.exp(-(cost1-cost0)/T)
        
            if cost1 < cost_min:
                x_min = x_new.copy()
                cost_min = cost1
#                print( '{:>7.3f} {}'.format(cost_min, x_min.loc[x_min['115590104078']>0,['115590104078']].index.values) )
                
            # Is it better, or does it make the probability
            # cutoff?
            if (cost1<cost0 or random.random()<p):
                cost0 = cost1
                x = x_new.copy()
            
        # Decrease the temperature
        T=T*cool
#        print (T)
    
    print('minimal cost is ',cost_min)
#    print(x_min)
    return x_min,cost_min




######################################################################    
#    for OrderID in OrderIDlist.drop_duplicates():
#        Order = OrderFile.loc[OrderFile[u'单据编号']==OrderID,:]
#        if Order.shape[0]>1:
#            goods = Order[u'商品代码']
#            if not goods.is_unique:
#                print ('not unique')
#            break
#goodsIDnotUnique = 'xp20180808643621'
OrderID = 'xp20180808381863'

if __name__ == '__main__' :
    
    t1 = time.time() 
    try:
        init += 1
    except:
        init = 0
        print ('data loading...')
        
        ###########
        warehouseFile = pd.DataFrame(pd.read_excel(r'仓库档案.xlsx'))
        warehouse = warehouseFile[u'仓库代码']
        warehouseDataframe = warehouseFile.drop([u'仓库代码',], axis = 1,)
        warehouseDataframe = pd.DataFrame(warehouseDataframe.values, index = warehouse, columns = warehouseDataframe.columns)
        ###########
        storeFile = pd.DataFrame(pd.read_excel(r'库存数据.xlsx'))
        ###########
        OrderFile = pd.DataFrame(pd.read_excel(r'零售小票.xlsx'))
        OrderIDlist = OrderFile[u'单据编号']
        print ('loading complete')
        ###########
        with open('全国各区经纬度.csv',encoding='UTF-8') as f1:
            disatnceFile = pd.read_csv(f1)
    
    t2 = time.time()
    x_all = [] 
    OrderID_all = []
    cost_min = [] 
#    
    error = []
    sku = []  
    #########################################################################
    for OrderID in OrderIDlist[:10000].drop_duplicates():
        try:
            Order = OrderfromTable(OrderID, OrderFile,)
            goodsTable, address = OrderInfo(Order)
            storeNum = stroeTable(goodsTable.index,warehouse, storeFile,)
            priority = warehouseDataframe.loc[warehouse,u'仓库优先级']
            distances = disGet(disatnceFile, warehouseDataframe, address)
            x,variableID = initially(goodsTable,storeNum)
            x,cost = annealingoptimize(x,storeNum,distances,priority,variableID,T=10.0,cool=0.95,iterMax=10)
            x_all.append(x)
            OrderID_all.append(OrderID)
            cost_min.append(cost)
            sku.append(list(goodsTable.index))
        except:
            error.append(OrderID)
    t3 = time.time()
    interval1 = t3-t1
    interval2 = t3-t2
#        
    result_10000 = pd.DataFrame(cost_min,index=OrderID_all,columns=['min_cost'])
    
    result_10000.to_excel('result.xlsx')
    
    result_x = x_all
    for j in range(len(x_all)):
        result_x[j] = result_x[j].values.T
#        
    result_10000[u'商品代码'] = sku
    result_10000[u'发货情况'] = x_all
    result_10000.to_excel('result_10000.xlsx')
    #########################################################################
    
    
#    x_sel = x_all
#    for i in range(len(x_sel)):
#        x_sel[i] = csr_matrix(x_sel[i])
#        li = list(x_sel[i].nonzero())
#        li.append(x_sel[i].data)
#        li = np.array(li,dtype='int')
#        x_sel[i] = li
    
#    result_x = pd.DataFrame(x_all,index=OrderID_all,columns=['仓库发货序列']) 
#    result_x.to_excel('result_x.xlsx')
#    re = x_all[0:50]
#    for i in range(50):
#        re[i] = re[i].values.T
#    re = pd.DataFrame(re)
#    re.to_excel('re.xlsx')
#    np.where(re[37])[1]
    
#   #    
    
#[2.0,
# 4.885706152746678,
# 3.0,
# 9.048107141908131,
# 6.298378389451399,
# 3.0,
# 7.252576302814183,
# 7.90979632978802,
# 34.5366101762726,
# 3.0,
# 4.0,
# 6.818102198288768,
# 3.0,
# 4.885706152746678,
# 5.0,
# 4.0,
# 6.07148172711478,
# 4.0,
# 4.0,
# 5.0,
# 7.405783939448559,
# 5.0,
# 10.120367636016647,
# 5.0,
# 5.0,
# 3.0,
# 6.07148172711478,
# 6.298378389451399,
# 3.0,
# 3.0,
# 3.0,
# 3.0,
# 3.0,
# 7.0,
# 5.0,
# 2.0,
# 2.0,
# 2.0,
# 5.737480593538514,
# 5.548709139896838,
# 5.0,
# 7.2793106921559225,
# 6.298378389451399,
# 3.0,
# 4.0,
# 3.0,
# 3.0,
# 5.0,
# 16.804008515361325,
# 2.0]    
    
######################问题货物解决#############################################    
#    error_all = []
#    error_x = []
#    error_sku = []
#    for OrderID in error:    
#        try:
#            Order = OrderfromTable(OrderID, OrderFile,)
#            goodsTable, address = OrderInfo(Order)
#            storeNum = stroeTable(goodsTable.index,warehouse, storeFile,)
#            priority = warehouseDataframe.loc[warehouse,u'仓库优先级']
#            distances = disGet(disatnceFile, warehouseDataframe, address)
#            x, variableID = initially(goodsTable,storeNum)
#            x,cost = annealingoptimize(x,storeNum,distances,priority,variableID,T=10000.0,cool=0.95)
#            error_sku.append(list(goodsTable.index))
#        #    x_all.append(x)
#        #    OrderID_all.append(OrderID)
#        #    cost_min.append(cost)
#            error_all.append(cost)
#            error_x.append(x)
#        except Exception as e:
#            error_all.append( str(e))
#            error_x.append(x)
#            
#    for i in range(len(error_x)):
#        error_sku.append(list(error_x[0].columns) )       
#    err_result = pd.DataFrame(error_all,index=error,columns=['min_cost'])
#    for j in range(len(error_x)):
#        error_x[j] = error_x[j].values.T
#    err_result[u'商品代码'] = error_sku
#    err_result[u'发货情况'] = error_x
#    err_result.to_excel('err.xlsx')
######################问题货物解决#############################################     
    
    
#    x_train = np.array(x_all[0])
#    xx = csr_matrix(x_train)
#    print(xx)
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    