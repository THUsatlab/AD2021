- # NCMMSC2021阿尔茨海默症识别比赛基线（短音频赛道）

  

  ## 系统描述

  本基线系统基于CNN，用Pytorch实现，主要包括2个核心脚本:

  - ```
    train.py
    ```

    - 本脚本主要是使用训练集训练出AD分类模型，并计算模型在验证集上的识别率。
    - 训练出的模型默认存储在`../data/model`文件夹下。


  - ```
    test.py
    ```

    - 本脚本主要是生成测试集预测的标签，并存储为txt格式。
    - txt文件存储在`results/`文件夹下。

  ## 使用方式

  ### 1. 下载基线系统

  可以从GitHub下载本基线系统，[AD2021/ncmmsc2021_baseline_cnn_pytorch at main · THUsatlab/AD2021 (github.com)](https://github.com/THUsatlab/AD2021/tree/main/ncmmsc2021_baseline_cnn_pytorch)。

  ### 2. 下载数据集

  联系主办方获取训练集和测试集数据。

  ### 3. 解压基线系统和数据集

  获取基线系统和数据集后，解压。基线系统和数据集的文件目录结构如下：

  - /ncmmsc2021_baseline_cnn_pytorch

    - /train.py
    - /test.py
    - /utility.py
    - /preprocess.py
    - /models.py
    - /data.py
    - /readme.md
    - /requirements.txt
  - /data
    - /train_audio
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

    - /test_audio (**后续发布测试数据**)
      - /F_030809_001.wav
      - /F_050005_001.wav
      - ...
      - /M_240706_001.wav
      - /M_242607_001.wav
      - /F_010202_001.wav
      
    - /train_melspec
    - /train_spec
    - /train_mfcc
    - /test_melspec
    - /test_spec
    - /test_mfcc
    - /model
    - /result

  ### 4. 修改参数

  参赛方可自行修改基线系统中的参数进行训练。

  ### 5. 在训练集上运行训练脚本

  运行预处理数据脚本`preprocess.py`。

  ```
  $ python preprocess.py
  ```

  运行训练脚本`train.py`。

  ```
  $ python train.py 
  ```

  `train.py` 训练出AD分类模型，并存储在  `model/`文件夹中，并对划分出的验证集进行预测，将3类的识别率的算术平均值作为整体的识别率。

  ### 6. 在测试集上运行测试脚本 

  运行预处理数据脚本`preprocess.py`。

  ```
  $ python preprocess.py --train_test test
  ```

  运行测试脚本`test.py`. 

  ```
  $ python3 test.py 
  ```

   `test.py` 使用训练出的分类模型，预测测试集每条数据的标签，并存储为txt文件。其中各标签的定义如下：1对应AD，2对应MCI，3对应HC。

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
  
  我们在 Windows、 Linux 和 MacOS 系统平台上开发的源码。主要依赖的软件包如下,详细见requirements.txt：
  
  ### Software packages
  
  - Python>=3.8.5
  
  ### Python packages
  
  - librosa>=0.8.1
  - numpy>=1.21.0
  - scikit-learn>=0.24.2
  - torch>=1.9.0+cu102
  - matplotlib>=3.4.2

