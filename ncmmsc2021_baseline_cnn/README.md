- # NCMMSC2021阿尔茨海默症识别比赛基线（短音频赛道）

  

  ## 系统描述

  本基线系统基于CNN，主要包括2个核心脚本:

  - ```
    train_ad.py
    ```

    - 本脚本主要是使用训练集训练出AD分类模型，并计算模型在验证集上的识别率和F1值。
    - 训练出的模型存储在`model/baseline`文件夹下。
    - 对参赛方自己划分出的验证集进行预测，根据预测结果分别计算AD、MCI、HC 3类数据的识别率和F1值，将3类的识别率的算术平均值作为整体的平均值，将3类的F1值的算术平均值作为整体的F1值。

  - ```
    test_ad.py
    ```

    - 本脚本主要是生成测试集预测的标签，并存储为txt格式。
    - txt文件存储在`results/`文件夹下。

  ## 使用方式

  ### 1. 下载基线系统

  可以从GitHub下载本基线系统，[AD2021/ncmmsc2021_baseline_cnn at main · THUsatlab/AD2021 (github.com)](https://github.com/THUsatlab/AD2021/tree/main/ncmmsc2021_baseline_cnn)。

  ### 2. 下载数据集

  联系主办方获取训练集和测试集数据。

  ### 3. 解压基线系统和数据集

  获取基线系统和数据集后，解压。基线系统和数据集的文件目录结构如下：

  - /ncmmsc2021_baseline_cnn

    - /train_ad.py
    - /test_ad.py
    - /utility.py
    - /test_utility.py
    - /data_processing.py
    - /test_processing.py
    - /models.py
    - /eval.py
    - /readme.md
    - /requirements
    - /genre_label_map.json
  - /data
    - /traindata
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

    - /testdata (**后续发布测试数据**)

      - /F_030809_001.wav
      - /F_050005_001.wav
      - ...
      - /M_240706_001.wav
      - /M_242607_001.wav
      - /F_010202_001.wav
      
    - /label
    - /pickles
    - /results
    - /train
        - / mfcc
        - / spec
        - / melspec
    - /test
        - / mfcc
        - / spec
        - / melspec  
    
    - /models
        - / baseline
            - / mfcc
            - / spec
            - / melspec

  ### 4. 修改参数

  参赛方可自行修改基线系统中的参数进行训练。

  ### 5. 在训练集上运行训练脚本

  运行训练脚本 train_ad.py。

  ```
  $ python3 train_ad.py 
  ```

  `train_ad.py` 训练出AD分类模型，并存储在  `model/`文件夹中，并对划分出的验证集进行预测，根据预测结果分别计算AD、MCI、HC 3类数据的识别率和F1值，将3类的识别率的算术平均值作为整体的识别率，将3类的F1值的算术平均值作为整体的F1值。

  ### 6. 在测试集上运行测试脚本 

  运行测试脚本`test_ad.py`. 

  ```
  $ python3 test_ad.py 
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
  
  我们在 Ubuntu 16.04 LTS 和 18.04 LTS 系统平台上开发的源码。主要依赖的软件包如下,详细见requirements.txt：
  
  ### Software packages
  
  - Python == 3.7.10
  - librosa == 0.8.0
  
  ### Python packages
  
  - Keras == 2.3.1
  - Keras-Applications == 1.0.8
  - Keras-Preprocessing == 1.1.0
  - matplotlib == 3.4.1
  - numpy == 1.16.2
  - PyYAML == 5.3.1
  - scikit-learn == 0.24.1
  - scipy == 1.5.4
  - tensorflow == 1.15.0
  - tqdm == 4.59.0

