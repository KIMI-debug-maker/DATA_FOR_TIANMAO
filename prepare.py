import gc
import pandas as pd
import time
paths = './data_format1/data_format1'
print(time.time())
data = pd.read_csv(f'{paths}/user_log_format1.csv', dtype={'time_stamp':'str'})
print(time.time())
data1 = pd.read_csv(f'{paths}/user_info_format1.csv')
print(time.time())
data2 = pd.read_csv(f'{paths}/train_format1.csv')
print(time.time())
submission = pd.read_csv(f'{paths}/test_format1.csv')
print(time.time())
data_train = pd.read_csv('./data_format2/data_format2/train_format2.csv')
print(time.time())
data2['origin'] = 'train'
submission['origin'] = 'test'
matrix = pd.concat([data2, submission], ignore_index=True, sort=False)
matrix.drop(['prob'], axis=1, inplace=True)
matrix = matrix.merge(data1, on='user_id', how='left')
data.rename(columns={'seller_id':'merchant_id'}, inplace=True)
data['user_id'] = data['user_id'].astype('int32')
data['merchant_id'] = data['merchant_id'].astype('int32')
data['item_id'] = data['item_id'].astype('int32')
data['cat_id'] = data['cat_id'].astype('int32')
data['brand_id'].fillna(0, inplace=True)#缺失值
data['brand_id'] = data['brand_id'].astype('int32')
data['time_stamp'] = pd.to_datetime(data['time_stamp'], format='%H%M')
matrix['age_range'].fillna(0, inplace=True)#缺失值
matrix['gender'].fillna(2, inplace=True)#缺失值
matrix['age_range'] = matrix['age_range'].astype('int8')
matrix['gender'] = matrix['gender'].astype('int8')
matrix['label'] = matrix['label'].astype('str')
matrix['user_id'] = matrix['user_id'].astype('int32')
matrix['merchant_id'] = matrix['merchant_id'].astype('int32')

del data1, data2
gc.collect()

#连接user_id特征，这些特征代表user在所有商家进行购物的统计数据
groups = data.groupby(['user_id'])
# u1 表示总操作次数
temp = groups.size().reset_index().rename(columns={0:'u1'})
matrix = matrix.merge(temp, on='user_id', how='left')
# u2 表示商品总数
temp = groups['item_id'].agg([('u2', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
# u3表示类别总数
temp = groups['cat_id'].agg([('u3', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
# u4表示商家总数
temp = groups['merchant_id'].agg([('u4', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
# u5表示品牌总数
temp = groups['brand_id'].agg([('u5', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
# u6表示时间，为浮点数
temp = groups['time_stamp'].agg([('F_time', 'min'), ('L_time', 'max')]).reset_index()
temp['u6'] = (temp['L_time'] - temp['F_time']).dt.seconds/3600
matrix = matrix.merge(temp[['user_id', 'u6']], on='user_id', how='left')
# u7,u8,u9,u10为点击方式
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'u7', 1:'u8', 2:'u9', 3:'u10'})
matrix = matrix.merge(temp, on='user_id', how='left')


#连接merchant_id特征，这些特征代表商家对所有user的销售统计数据
groups = data.groupby(['merchant_id'])

# m1 为在该商家操作数
temp = groups.size().reset_index().rename(columns={0:'m1'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
# m2为看过该商家的不同人个数，m3为该商家被看过的不同产品数，m4为该商家涵盖的类别总数，m5为不同品牌数
temp = groups['user_id', 'item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'user_id':'m2',
    'item_id':'m3',
    'cat_id':'m4',
    'brand_id':'m5'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
# m6,m7,m8,m9同为该商家的点击数情况
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'m6', 1:'m7', 2:'m8', 3:'m9'})
matrix = matrix.merge(temp, on='merchant_id', how='left')


# 连接action为-1的情况，m10记作action未知的个数
temp = data_train[data_train['label'] == -1].groupby(['merchant_id']).size().reset_index().rename(columns={0:'m10'})
matrix = matrix.merge(temp, on='merchant_id', how='left')


# user-merchant特征，这些特征代表一个user在一个商家进行一笔购物时，对应的购物信息
groups = data.groupby(['user_id', 'merchant_id'])
# um1 一个user在一个店家购买量
temp = groups.size().reset_index().rename(columns={0:'um1'})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
# um2，um3，um4分别是购买的产品量，类别量和品牌量
temp = groups['item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'item_id': 'um2',
    'cat_id': 'um3',
    'brand_id': 'um4'
})
# um5,um6,um7,um8为点击量情况
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={
    0:'um5',
    1:'um6',
    2:'um7',
    3:'um8'
})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
# um9为时间特征
temp = groups['time_stamp'].agg([('frist', 'min'), ('last', 'max')]).reset_index()
temp['um9'] = (temp['last'] - temp['frist']).dt.seconds/3600
temp.drop(['frist', 'last'], axis=1, inplace=True)
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')


matrix['r1'] = matrix['u9']/matrix['u7'] #用户购买点击比
matrix['r2'] = matrix['m8']/matrix['m6'] #商家购买点击比
matrix['r3'] = matrix['um7']/matrix['um5'] #不同用户不同商家购买点击比

matrix.fillna(0, inplace=True)#缺失值

#train, test_setdata
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)

train_data.to_pickle('./train_data_filled.pkl')
test_data.to_pickle('./test_data_filled.pkl')
