import os
from compare_model import *
from SANet import *
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import tensorflow as tf
from data import *
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
model_name_list = ['UNet', 'DeepUNet', 'SegNet', 'DeepLabv3+', 'SANet']
#====================0=========1==========2===========3===========4==============
# ==================================set prama=======================================
model_name = model_name_list[4]
TRAIN_IMAGE_SIZE = 256
TEST_IMAGE_SIZE = 1024
epochs = 100
batch_size = 8
# =============================================================================
if model_name=='UNet':
    model = UNet(input_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 4), Falg_summary=True, Falg_plot_model=False)
if model_name == 'DeepUNet':
    model = DeepUNet(input_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 4), Falg_summary=True, Falg_plot_model=False)
if model_name == 'SegNet':
    model = SegNet(input_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 4), Falg_summary=True, Falg_plot_model=False)
if model_name == 'DeepLabv3+':
    model = Deeplabv3(input_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 4), Falg_summary=True, Falg_plot_model=False)
if model_name == 'SANet':
    model = SANet(input_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 4), Falg_summary=True, Falg_plot_model=False)
# =============================================================================
savePath = mkSaveDir(model_name)
checkpointPath= savePath + "/"+model_name+"-{epoch:03d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpointPath, monitor='val_acc', verbose=1,
                             save_best_only=False, save_weights_only=True, mode='auto', period=1)
EarlyStopping = EarlyStopping(monitor='val_acc', patience=200, verbose=1)
tensorboard = TensorBoard(log_dir=savePath, histogram_freq=0)
callback_lists = [tensorboard, EarlyStopping, checkpoint]
# ==============================================================================
train_image, train_GT, valid_image, valid_GT = readNpy()   #read data
History = model.fit(train_image, train_GT, batch_size=batch_size, validation_data=(valid_image, valid_GT),
    epochs=epochs, verbose=1, shuffle=True, class_weight='auto', callbacks=callback_lists)
with open(savePath + '/log.txt','w') as f:
    f.write(str(History.history))




