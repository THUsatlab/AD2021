########################################################################
# 用于发布测试集后，生成要提交的测试结果
########################################################################
from test_processing import check_size
from test_processing import create_dataset
from test_processing import save_feature_test
from eval import load_test_data
########################################################################
DATA_PATH = '../data/'
AUDIO_ROOT_PATH = DATA_PATH + 'testdata/' 
SPLITS_FOLDER = DATA_PATH + 'label/'
MELSPECS_FOLDER = DATA_PATH + 'train/melspec/'
SPECS_FOLDER = DATA_PATH + 'train/spec/'
MFCCS_FOLDER = DATA_PATH + 'train/mfcc/'

MELSPECS_FOLDER_test = DATA_PATH + 'test/melspec/'
SPECS_FOLDER_test = DATA_PATH + 'test/spec/'
MFCCS_FOLDER_test = DATA_PATH + 'test/mfcc/'

MODELS_FOLDER = DATA_PATH + 'models/baseline/'
RESULTS_FOLDER = DATA_PATH + 'results/'
PICKLES_FOLDER = DATA_PATH + 'pickles/'
# 主函数
# 提取测试集的特征，根据训练的模型，得到预测的结果标签
if __name__ == "__main__":
    check_size(dataset_path=AUDIO_ROOT_PATH, compute_spec=True, num_segments=2)
    check_size(dataset_path=AUDIO_ROOT_PATH, compute_melspec=True)
    create_dataset(dataset_path=AUDIO_ROOT_PATH, melspecs_folder=MELSPECS_FOLDER_test, mfccs_folder=MFCCS_FOLDER_test,specs_folder = SPECS_FOLDER_test,num_segments=2)
    save_feature_test(DATA_PATH=DATA_PATH,MELSPECS_FOLDER=MELSPECS_FOLDER_test,SPECS_FOLDER=SPECS_FOLDER_test,MFCCS_FOLDER=MFCCS_FOLDER_test,SPLITS_FOLDER=SPLITS_FOLDER)
    load_test_data(DATA_PATH=DATA_PATH,SPLITS_FOLDER=SPLITS_FOLDER,MELSPECS_FOLDER=MELSPECS_FOLDER,
                            MELSPECS_FOLDER_test=MELSPECS_FOLDER_test,SPECS_FOLDER=SPECS_FOLDER,
                            SPECS_FOLDER_test=SPECS_FOLDER_test,MFCCS_FOLDER=MFCCS_FOLDER,
                            MFCCS_FOLDER_test=MFCCS_FOLDER_test,NUM_CLASSES = 3,NUM_EPOCHS = 30,BATCH_SIZE = 128,
                            MODELS_FOLDER=MODELS_FOLDER,RESULTS_FOLDER=RESULTS_FOLDER)  
    print('Test results are completed !')   
