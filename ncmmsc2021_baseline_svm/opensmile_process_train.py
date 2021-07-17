# -*- coding: utf-8 -*-
#该脚本和traindata文件夹位于同一级目录下
#./opensmile_fea为opensmile提取的训练集特征存储路径
#内含AD、HC、MCI 3种的特征
#提取后的特征存储在 opensmile_train文件夹下
#后续使用fea_process脚本进行规范化

import pandas as pd
import os
filepath=r'./opensmile_train'

for fea in os.listdir(filepath):
	fn=os.path.join(filepath,fea)
	df=pd.read_csv(fn,header=None,names=['name']+[str(i) for i in range(1,1584)])
	del df['1583']
	df.insert(1,'label',0)
	filename,extension = os.path.splitext(fea)
	#1对应AD，2对应MCI，3对应HC
	if filename=='AD':
		df['label']=1
	elif filename=='MCI':
		df['label']=2
	elif filename=='HC':
		df['label']=3
	else:
		print('Name Error!')
	#df = df.rename(columns=lambda x: x.replace("'","").replace('"','')).replace(" ","")
	df['name']=df['name'].apply(lambda x: x.strip("'"))
	os.remove(fn)
	df.to_csv(fn,index=False)

file_list=os.listdir(filepath)
df = pd.read_csv(os.path.join(filepath,file_list[0]))
#将读取的第一个CSV文件写入合并后的文件保存
df.to_csv(r'./opensmile_train/train.csv',index=False)
 
#循环遍历列表中各个CSV文件名，并追加到合并后的文件
for i in range(1,len(file_list)):
    df = pd.read_csv(os.path.join(filepath,file_list[i]))
    df.to_csv(r'./opensmile_train/train.csv',index=False, header=False, mode='a+')

