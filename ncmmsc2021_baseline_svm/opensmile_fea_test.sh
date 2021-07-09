#!/bin/sh

#该脚本和testdata测试集文件夹位于同一级目录下
#opensmile为opensmile工具安装路径
#dataset为测试集路径，testdata下为音频文件
#提取后的特征存储在 opensmile_test文件夹下
#后续使用opensmile_process脚本进行规范化
opensmile=/home/gy/opensmile-2.3.0
dataset=/home/gy/DATA/test_none_label
test='test'
c='.csv'
cd ${dataset}

row=`ls -l | wc -l`
fea=${test}${c}
fea_n=${test}'_new'${c}
for m in *.wav
do
	${opensmile}/SMILExtract -C "${opensmile}/config/IS10_paraling_compat.conf"  -I "${m}"  -O /home/gy/DATA/opensmile_test/${fea_n}  -instname "${m%.wav}"
done
tail -n ${row} /home/gy/DATA/opensmile_test/${fea_n} >/home/gy/DATA/opensmile_test/${fea}
rm /home/gy/DATA/opensmile_test/${fea_n}

python3 /home/gy/DATA/opensmile_process_test.py


