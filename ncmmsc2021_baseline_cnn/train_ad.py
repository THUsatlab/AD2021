########################################################################
# 导入所需模块，以及从common.py中导入计算准确率等指标的函数
########################################################################
from data_processing import check_size
from data_processing import create_dataset
from data_processing import save_feature_train
from models import train_model
########################################################################
#路径
DATA_PATH = '../data/' 
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
                               PICKLES_FOLDER=RESULTS_FOLDER,NUM_CLASSES = 3,NUM_EPOCHS = 30,BATCH_SIZE = 128)
    print('Train results are completed !')

