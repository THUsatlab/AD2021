########################################################################
# 导入所需模块，以及从common.py中导入计算准确率等指标的函数
########################################################################
from data_processing import check_size
from data_processing import create_dataset
from data_processing import save_feature_train
from models import train_model
import os
########################################################################
#路径
DATA_PATH = '../data/' 
def make_dirs(path):
    if not os.path.exists(path+'traindata/'):
        os.makedirs(path+'traindata/')
    if not os.path.exists(path+'train/melspec/'):
        os.makedirs(path+'train/melspec/')
    if not os.path.exists(path+'train/spec/'):
        os.makedirs(path+'train/spec/')
    if not os.path.exists(path+'train/mfcc/'):
        os.makedirs(path+'train/mfcc/')
    if not os.path.exists(path+'label/'):
        os.makedirs(path+'label/')
    if not os.path.exists(path+'models/baseline/'):
        os.makedirs(path+'models/baseline/')
    if not os.path.exists(path+'results/'):
        os.makedirs(path+'results/')
    if not os.path.exists(path+'pickles/'):
        os.makedirs(path+'pickles/')
    if not os.path.exists(path+'models/baseline/melspec/'):
        os.makedirs(path+'models/baseline/melspec/')
    if not os.path.exists(path+'models/baseline/mfcc/'):
        os.makedirs(path+'models/baseline/mfcc/')   
    if not os.path.exists(path+'models/baseline/spec/'):
        os.makedirs(path+'models/baseline/spec/')     
make_dirs(DATA_PATH)
AUDIO_ROOT_PATH = DATA_PATH + 'traindata/' 
MELSPECS_FOLDER = DATA_PATH + 'train/melspec/' 
SPECS_FOLDER = DATA_PATH + 'train/spec/' 
MFCCS_FOLDER= DATA_PATH + 'train/mfcc/' 
SPLITS_FOLDER = DATA_PATH + 'label/' 
MODELS_FOLDER = DATA_PATH + 'models/baseline/'
RESULTS_FOLDER = DATA_PATH + 'results/'
PICKLES_FOLDER = DATA_PATH + 'pickles/'
# 主函数
# 使用特征进行训练，可根据选择的模型自行划分训练集、验证集，训练完成后保存模型
if __name__ == "__main__":
    check_size(dataset_path=AUDIO_ROOT_PATH, compute_spec=True, num_segments=2)
    check_size(dataset_path=AUDIO_ROOT_PATH, compute_melspec=True)
    create_dataset(dataset_path=AUDIO_ROOT_PATH, melspecs_folder=MELSPECS_FOLDER, mfccs_folder=MFCCS_FOLDER,specs_folder = SPECS_FOLDER,num_segments=2)
    save_feature_train(DATA_PATH=DATA_PATH,MELSPECS_FOLDER=MELSPECS_FOLDER,SPECS_FOLDER=SPECS_FOLDER,MFCCS_FOLDER=MFCCS_FOLDER,SPLITS_FOLDER=SPLITS_FOLDER)
    train_model(DATA_PATH=DATA_PATH,SPLITS_FOLDER=SPLITS_FOLDER,
                               MELSPECS_FOLDER=MELSPECS_FOLDER,SPECS_FOLDER=SPECS_FOLDER,
                               MFCCS_FOLDER=MFCCS_FOLDER,MODELS_FOLDER=MODELS_FOLDER,RESULTS_FOLDER=RESULTS_FOLDER,
                               PICKLES_FOLDER=PICKLES_FOLDER,NUM_CLASSES = 3,NUM_EPOCHS = 30,BATCH_SIZE = 128)
    print('Train results are completed !')

