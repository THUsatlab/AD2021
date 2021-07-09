- # NCMMSC2021阿尔茨海默症识别比赛基线(长音段赛道)

  

  ## 系统描述
  基线系统主要包3部分核心脚本:

  - ```
    opensmile_fea_test.sh
	opensmile_process_test.py
	opensmile_fea_train.sh
	opensmile_process_train.py
    ```

    - 本部分脚本主要是使用opensmile工具进行批量提取音频特征并对提取完的特征文件做规范化处理。

  - ```
    train_ad.py 
    ```
	- 本脚本主要是将训练集分别为训练集和验证集，使用训练集训练分类器，并进行五次交叉验证，
	  最后给出五次交叉验证的识别率，查准率，召回和F1值的平均值，模型没有保存，因为后面获
	  得测试集后需要将训练集和测试集的特征进行统一归一化，重新训练，因此本脚本只是用于获得训练集后熟悉流程。
	
  - ```
    test_ad.py
    ```
    - 本脚本主要是使用训练集训练出AD分类模型，并计算模型在验证集上的识别率，查准率，召回和F1值（采用sklearn.metrics自带的函数进行估计）。
    - 并且本脚本还可以用来生成测试集预测的标签，并存储为txt格式。
    - txt文件存储在`result/`文件夹下。

  ## 使用方式

  ### 1. 下载基线系统

  可以从GitHub下载本基线系统。

  ### 2. 下载数据集

  联系主办方获取训练集和测试集数据。

  ### 3. 解压基线系统和数据集

  获取基线系统和数据集后，解压。基线系统和数据集的文件目录结构如下：

  - /NCMMSC2021_baseline_svm

    - /result
    - /feature
    - /train_ad.py
	- /test_ad.py
    - /opensmile_fea_test.sh
	- /opensmile_fea_train.sh
	- /opensmile_process_test.py
	- /opensmile_process_train.py
    - /readme.md
    
  - /NCMMSC2021_AD_Recognition_traindata

    - /AD

      - /AD_F_030807_001.wav
      - /AD_F_040006_001.wav
      - ...
      - /AD_M_230706_001.wav
      - /AD_M_242608_001.wav
    - /HC 
      - /HC_F_019202_001.wav
      - ...
      - /HC_M_262520_001.wav
    - /MCI
      - /MCI_F_031912_001.wav
      - /MCI_M_252308_002.wav

  - /eval_data 

    - /F_030808_001.wav
    - ...
    - /M_230705_001.wav

  ### 4. 修改参数

  参赛方可自行修改基线系统中的参数进行训练。

  ### 5. 提取特征

  运行训练脚本 `opensmile_fea_train.sh`获得训练集特征数据，
  运行训练脚本 `opensmile_fea_test.sh`获得测试集特征数据
  
  ```
  最后处理后的格式如下所示:
   name	          label	    1        	2	       3     ...
   AD_F_030807_001	1	9.65E+02	1.79E+03	3.42E-01 ...
   AD_F_040006_001	1	5.89E+03	3.09E+03	2.47E-01 ...
   AD_F_040006_002	1	3.64E+03	1.45E+03	4.99E-01 ...
   AD_F_040108_001	1	1.69E+03	3.91E+03	4.91E-01 ...
   AD_F_040108_002	1	1.68E+03	5.11E+03	5.08E-01 ...
   AD_F_040108_003	1	2.80E+03	3.50E+03	5.08E-01 ...
   ....            ...     ...         ...        ...
  ```


  ### 6. 训练模型预测标签 
  
  运行训练脚本`train_ad.py`. 

  ```
  $ python train_ad.py 
  ``` 
  `train_ad.py` 拿到训练数据后熟悉整个流程。
  
  运行测试脚本`test_ad.py`. 

  ```
  $ python test_ad.py 
  ```

  `test_ad.py` 使用训练出的分类模型，预测测试集每条数据的标签，并存储为txt文件。其中各标签的定义如下：1对应AD，2对应MCI，3对应HC。

  `result.txt`中文件格式如下：第一列文件名（文件名不带路径，带wav后缀），第二列标签，各列使用1个空格分隔。

  ```
  F_030808_001.wav 1
  F_030809_001.wav 2
  F_030818_001.wav 3
  M_030821_001.wav 3
  M_030354_001.wav 2
  M_030763_001.wav 1
  ...
  ```

  ## 依赖库
  
  我们在 Ubuntu 16.04.6 LTS 系统平台上开发的源码。主要依赖的软件包如下：
  
  ### Software packages
  
  - Python == 3.6.12
  - opensmile == 2.3.0
  ### Python packages
  
  - numpy == 1.18.5
  - scikit-learn == 0.23.2
  - pandas == 1.1.2
  


