from data import *
from compare_model import *
import os
import numpy as np
import glob
from keras import backend as K
import time
from SANet import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
model_name_list = ['UNet', 'DeepUNet', 'SegNet', 'DeepLabv3+', 'SANet']
#====================0=========1==========2===========3===========4==============
testlogdir = 'logs/UNet-2020-12-03_22-21'
TEST_IMAGE_SIZE = 1024
model_name = model_name_list[0]
def modelTest(modelPath, i, IMAGE_SIZE, total, test_p, test_name, model_name, data_dir):
    '''
    load the weighted, predict the test results.
    '''
    # =============================================================================
    if model_name == 'UNet':
        model = UNet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 4), Falg_summary=False, Falg_plot_model=False,pretrained_weights=modelPath)
    if model_name == 'DeepUNet':
        model = DeepUNet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 4), Falg_summary=False, Falg_plot_model=False,pretrained_weights=modelPath)
    if model_name == 'SegNet':
        model = SegNet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 4), Falg_summary=False, Falg_plot_model=False,pretrained_weights=modelPath)
    if model_name == 'DeepLabv3+':
        model = Deeplabv3(input_size=(IMAGE_SIZE, IMAGE_SIZE, 4), Falg_summary=False, Falg_plot_model=False,pretrained_weights=modelPath)
    if model_name == 'SANet':
        model = SANet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 4), Falg_summary=False, Falg_plot_model=False,pretrained_weights=modelPath)
    # =============================================================================
    test_img = np.load(test_p + '/test_image_arr.npy')
    test_Gt = np.load(test_p + '/test_mask_arr.npy')
    # =============================================================================
    print('%d / %d' % (i, total))
    scores = model.evaluate(test_img, test_Gt, verbose=0)
    print('Test loss:', scores[0])
    print('Test accu:', scores[1])
    t1 = time.time()
    y_predict = model.predict(test_img)
    K.clear_session()
    # ==============================save_predict===============================================
    modelName = modelPath.split('/')[-1]
    modelName1 = modelName.split('.')[0]
    savePath = data_dir + '_test/' + test_name
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    imgsavepath = savePath + '/' + modelName1 + '-Accu-' + str(scores[1])[:6] + 'Pred.tif'
    t2 = time.time()
    print('time:', (t2 - t1))
    saveImg(y_predict, imgsavepath)
    print('save predict image')
    return t2 - t1

def test(data_dir,IMAGE_SIZE,model_name):

    f_names_hdf5 = glob.glob(data_dir + '/*.hdf5')
    f_names_hdf5 = f_names_hdf5[0:]

    tdirs = os.listdir('data')

    for file in tdirs:
        if file != 'train' and file != 'valid':
            test_p = os.path.join('data', file)
            tc = 0.0  # time consume
            tcs = 1
            for j, f_name in enumerate(f_names_hdf5):
                tc += modelTest(f_name, j + 1, IMAGE_SIZE, total=len(f_names_hdf5), test_p=test_p, test_name = file, data_dir = data_dir, model_name = model_name)
                tcs += 1
            tc = tc / tcs
            print('测试平均时间为：'+str(tc))
if __name__ == '__main__':
    test(testlogdir,TEST_IMAGE_SIZE,model_name)
