# -*- coding: utf-8 -*-
#该脚本和testdata测试集文件夹位于同一级目录下
#./opensmile_test为opensmile提取的测试集特征存储路径
#内含测试集的特征
#提取后的特征存储在 opensmile_test文件夹下
#后续使用fea_process脚本进行规范化

import pandas as pd
import os
filepath=r'/home/gy/DATA/opensmile_test'
fea=r'test.csv'

fn=os.path.join(filepath,fea)
df=pd.read_csv(fn,header=None,names=['name']+[str(i) for i in range(1,1584)])
del df['1583']

df['name']=df['name'].apply(lambda x: x.strip("'"))
os.remove(fn)
df.to_csv(fn,index=False)

