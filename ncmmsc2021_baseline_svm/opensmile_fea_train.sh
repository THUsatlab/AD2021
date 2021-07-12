#!/bin/sh

#该脚本和traindata文件夹位于同一级目录下
#opensmile为opensmile工具安装路径
#dataset为训练集路径，traindata下含AD、HC、MCI 3个文件夹
#提取后的特征存储在 opensmile_train文件夹下
#后续使用opensmile_process脚本进行规范化
opensmile=/home/gy/opensmile-2.3.0
dataset=/home/gy/DATA/train
c='.csv'
cd ${dataset}
for subdir in `ls ${dataset}`
do 
	cd ${dataset}'/'${subdir}
	row=`ls -l | wc -l`
	fea=${subdir}${c}
	fea_n=${subdir}'_new'${c}
	for m in *.wav
	do
		${opensmile}/SMILExtract -C "${opensmile}/config/IS10_paraling_compat.conf"  -I "${m}"  -O /home/gy/DATA/opensmile_train/${fea_n}  -instname "${m%.wav}"
	done
	line=`expr $row - 1`
	tail -n ${line} /home/gy/DATA/opensmile_train/${fea_n} >/home/gy/DATA/opensmile_train/${fea}
	rm /home/gy/DATA/opensmile_train/${fea_n}
done

python3 /home/gy/DATA/opensmile_process_train.py


