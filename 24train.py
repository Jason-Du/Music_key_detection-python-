from keras.models import Sequential,Model
from keras.layers import Input,Dense,Conv2D,Flatten,MaxPooling2D,Activation,Dropout,AveragePooling2D,ZeroPadding2D,concatenate,add,GlobalAveragePooling2D,Conv1D,MaxPooling1D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.initializers import Initializer
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data(data_dirname_path):
	Trainx = np.load(os.path.join(data_dirname_path,"Trainx.npy"))
	Trainy = np.load(os.path.join(data_dirname_path,"Trainy.npy"))
	Vaildx = np.load(os.path.join(data_dirname_path,"Vaildx.npy"))
	Vaildy = np.load(os.path.join(data_dirname_path,"Vaildy.npy"))
	Testx  = np.load(os.path.join(data_dirname_path,"Testx.npy"))
	Testy  = np.load(os.path.join(data_dirname_path,"Testy.npy"))
	print(Trainx.shape)


	Trainx =Trainx.reshape(-1,166,586,1)
	Vaildx = Vaildx.reshape(-1, 166, 586,1)
	Testx  = Testx.reshape(-1,166, 586, 1)


	Trainy  =    np_utils.to_categorical(Trainy,num_classes=24)
	Vaildy  =    np_utils.to_categorical(Vaildy,num_classes=24)
	Testy   =    np_utils.to_categorical(Testy,num_classes=24)


	return Trainx,Trainy,Vaildx,Vaildy,Testx,Testy
def DNN():
	# 创建模型
	model = Sequential()
	model.add(Dense(2, input_dim=(512*586), activation='relu'))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	# 查看网络结构
	model.summary()
	# 保存模型
	model.save('DNN.h5')
	return model

def VGG(width, height, depth, classes):
    # initialize the model
	model = Sequential()

	model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(5, 5), filters=32, strides=(1, 1), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=32, strides=(1, 1), activation='relu',padding="SAME"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=64, strides=(1, 2), activation='relu', padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=64, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=64, strides=(2, 1), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=64, strides=(1, 1), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=32, strides=(1, 1), activation='relu',padding="SAME"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=32, strides=(2, 2), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=32, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(5, 5), filters=16, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding="SAME"))
	model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(5, 5), filters=16, strides=(2, 2), activation='relu',padding="SAME"))
	model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(5, 5), filters=8, strides=(1, 1), activation='relu',padding="SAME"))

	# Fully connection layer
	model.add(Flatten())
	model.add(Dense(64,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))
	# model.save("VGG_12Tone_Major.h5")

	return model
def VGGCONV1(width, height, classes):
    # initialize the model
	model = Sequential()
	model.add(Conv1D(input_shape=(width,height), data_format="channels_first",kernel_size=(10), filters=46, strides=(10), activation='relu',padding="SAME"))
	model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=46, strides=(10), activation='relu',padding="SAME"))
	model.add(Dropout(0.5))
	model.add(Conv1D(data_format="channels_first", kernel_size=(2), filters=46, strides=(10), activation='relu', padding="SAME"))
	# model.add(MaxPooling1D(pool_size=(10), strides=(2), padding="SAME"))
	model.add(Conv1D(data_format="channels_first", kernel_size=(2), filters=92, strides=(1), activation='relu', padding="SAME"))
	model.add(Conv1D(data_format="channels_first", kernel_size=(2), filters=92, strides=(1), activation='relu',padding="SAME"))
	model.add(Dropout(0.5))
	model.add(Conv1D(data_format="channels_first", kernel_size=(2), filters=46, strides=(1), activation='relu',padding="SAME"))
	# model.add(MaxPooling1D(pool_size=(10), strides=(2), padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=46, strides=(1), activation='relu',padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=46, strides=(1), activation='relu',padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=46, strides=(1), activation='relu',padding="SAME"))
	# model.add(MaxPooling1D(pool_size=(10), strides=(2), padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=1024, strides=(2), activation='relu',padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=512, strides=(1), activation='relu',padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=512, strides=(1), activation='relu',padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=512, strides=(1), activation='relu',padding="SAME"))
	# model.add(MaxPooling1D(pool_size=(10), strides=(2), padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=256, strides=(2), activation='relu',padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=256, strides=(1), activation='relu',padding="SAME"))
	# model.add(Conv1D(data_format="channels_first", kernel_size=(10), filters=256, strides=(1), activation='relu',padding="SAME"))

	# Fully connection layer
	model.add(Flatten())
	model.add(Dense(46,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(46,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(23, activation='relu'))
	model.add(Dropout(0.5))
	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))
	return model

def AllConv(width, height, depth, classes):
    # initialize the model
	model = Sequential()

	# model.add(MaxPooling2D(input_shape=(width,height,depth),pool_size=(1, 2), strides=(1, 1), padding="SAME"))
	model.add(Conv2D(input_shape=(width, height, depth),kernel_size=(2,5), filters=4, strides=(1,2), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(2, 5), filters=4,strides=(1, 1), activation='relu', padding="SAME"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding="SAME"))
	model.add(Conv2D(data_format="channels_last",kernel_size=(2,5), filters=8, strides=(1,1), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(2, 5), filters=8,strides=(1, 1), activation='relu', padding="SAME"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding="SAME"))
	# model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(1, 5), filters=64, strides=(1, 5), activation='relu',padding="SAME"))
	# model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(1, 5), filters=128, strides=(1,5 ), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(2, 5), filters=16, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(Conv2D(data_format="channels_last", kernel_size=(2, 5), filters=16, strides=(1, 2), activation='relu',padding="SAME"))
	# model.add(Conv2D(data_format="channels_last",kernel_size=(2, 5), filters=32, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding="SAME"))
	model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(2, 5), filters=64, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(2, 5), filters=64, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(2, 5), filters=24, strides=(1, 2), activation='relu',padding="SAME"))
	model.add(AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding="SAME"))
	# model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="SAME"))
	# model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(1, 5), filters=128, strides=(1, 2), activation='relu',padding="SAME"))
	# model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(1, 5), filters=128, strides=(1, 1), activation='relu',padding="SAME"))
	# model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(1, 5), filters=128, strides=(1, 2), activation='relu',padding="SAME"))
	# model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(1, 5), filters=8, strides=(1, 2), activation='relu',padding="SAME"))





	# Fully connection layer
	model.add(Flatten())
	# model.add(Dense(512,activation = 'relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(512,activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(256, activation='relu'))
	# model.add(Dropout(0.5))
	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))
	model.save("ALLCONV.h5")

	return model



def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name,data_format="channels_last")(x)
    return x
def Inception(x, nb_filter_para):
	(branch1, branch2, branch3, branch4) = nb_filter_para
	branch1x1 = Conv2D(branch1[0], (1, 1), padding='same', strides=(1, 2),activation='relu', name=None)(x)

	branch3x3 = Conv2D(branch2[0], (1, 1), padding='same', strides=(1, 1),activation='relu', name=None)(x)
	branch3x3 = Conv2D(branch2[1], (1, 5), padding='same', strides=(1, 2), activation='relu',name=None)(branch3x3)

	branch5x5 = Conv2D(branch3[0], (1, 1), padding='same', strides=(1, 1), activation='relu',name=None)(x)
	branch5x5 = Conv2D(branch3[1], (1, 1), padding='same', strides=(1, 2), activation='relu',name=None)(branch5x5)

	branchpool = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)
	branchpool = Conv2D(branch4[0], (1, 1), padding='same', strides=(1, 1), activation='relu',name=None)(branchpool)

	x =concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)



	return x
def InceptionV1(width, height, depth, classes):

	input = Input(shape=(width,height,depth))

	x = Inception(input, [(8,), (16, 16), (16, 16), (8,)])  # Inception 3b 28x28x480
	x = Inception(x, [(16,), (32, 32), (32,32), (16,)])  # Inception 3a 28x28x256
	x = Conv2d_BN(x, 64, (1, 5), strides=(1, 2), padding='same')
	x = Conv2d_BN(x, 64, (1, 5), strides=(1, 2), padding='same')
	x = Conv2d_BN(x, 32, (1, 5), strides=(1, 2), padding='same')
	x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)
	x = Conv2d_BN(x, 32, (1, 5), strides=(1, 2), padding='same')
	x = Conv2d_BN(x, 16, (1, 5), strides=(1, 2), padding='same')
	x = Conv2d_BN(x, 16, (1, 5), strides=(1, 2), padding='same')
	x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)
	# x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(classes, activation='softmax')(x)
	model = Model(input=input, output=x)
	model.save("INCEPTION_12Tone_Major.h5")



	return model
def Residual_Block(input_model, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
	x = Conv2d_BN(input_model, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')

	# need convolution on shortcut for add different channel
	if with_conv_shortcut:
		shortcut = Conv2d_BN(input_model, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, input_model])
		return x

def ResNet(width, height, depth, classes):
	input=Input(shape=(width, height, depth))
	x = Conv2d_BN(input, 16, (2, 5), strides=(1, 2), padding='same')
	x = Conv2d_BN(input, 16, (2, 5), strides=(1, 2), padding='same')
	x = MaxPooling2D(pool_size=(1, 5), strides=(1, 2), padding='same')(x)

	# Residual conv2_x ouput 56x56x64
	x = Residual_Block(x, nb_filter=32, kernel_size=(2, 5),strides=(1, 2),with_conv_shortcut=True)
	x = Residual_Block(x, nb_filter=32, kernel_size=(2, 5))
	x = MaxPooling2D(pool_size=(1, 5), strides=(1, 2), padding='same')(x)

	x = Residual_Block(x, nb_filter=64, kernel_size=(2, 5), strides=(1, 2),with_conv_shortcut=True)  # need do convolution to add different channel
	x = Residual_Block(x, nb_filter=64, kernel_size=(2, 5))
	x = Residual_Block(x, nb_filter=64, kernel_size=(2, 5))
	x = MaxPooling2D(pool_size=(1, 5), strides=(1, 2), padding='same')(x)

	# Residual conv4_x ouput 14x14x256
	x = Residual_Block(x, nb_filter=32, kernel_size=(2, 5), strides=(1, 2),with_conv_shortcut=True)  # need do convolution to add different channel
	x = Residual_Block(x, nb_filter=32, kernel_size=(2, 5))
	x = Residual_Block(x, nb_filter=32, kernel_size=(2, 5))
	x = MaxPooling2D(pool_size=(1, 5), strides=(1, 2), padding='same')(x)

	# # Residual conv5_x ouput 7x7x512
	x = Residual_Block(x, nb_filter=16, kernel_size=(2, 5), strides=(1, 2), with_conv_shortcut=True)
	x = Residual_Block(x, nb_filter=16, kernel_size=(2, 5))
	x = MaxPooling2D(pool_size=(1, 5), strides=(1, 2), padding='same')(x)

	# Using AveragePooling replace flatten
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(classes, activation='softmax')(x)

	model = Model(input=input, output=x)

	return model


if __name__ == '__main__':
	print()
	Initializer()
	# Read DATA
	npy_dirname="24Tone_cut_npy_file"
	csv_dirname="24Tone_cut_csv_file"
	data_npy_path=os.path.join(os.path.dirname(__file__),npy_dirname)
	data_csv_path=os.path.join(os.path.dirname(__file__),csv_dirname)
	Trainx,Trainy,Vaildx,Vaildy,Testx,Testy=read_data(data_npy_path)

	# Model Choose
	Model_name="ResNet"

	# DNN TRAIN
	# return Trainx,Trainy,Vaildx,Vaildy,Testx,Testy
	# RESHAPE SEQUENTIAL
	if Model_name=="DNN":


		model=DNN()
		model.compile(optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
		History = model.fit(Trainx, Trainy, batch_size=64, epochs=20, verbose=2, validation_data=(Vaildx, Vaildy))
		pre = model.evaluate(Testx, Testy, batch_size=64, verbose=2)
		print('test_loss:', pre[0], '- test_acc:', pre[1])

		plt.figure(figsize=(15, 5))
		plt.subplot(1, 2, 1)
		plt.plot(History.history['accuracy'])
		plt.plot(History.history['val_accuracy'])
		plt.title('DNN accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.subplot(1, 2, 2)
		plt.plot(History.history['loss'])
		plt.plot(History.history['val_loss'])
		plt.title('DNN loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		plt.show()
	elif Model_name == "VGG":
		model = VGG(128,586,1, 24)
		print(Trainx.shape)
		print(Testx.shape)
		print(Vaildx.shape)
		print(Trainy.shape)
		print(Vaildy.shape)
		print(Testy.shape)
		model.summary()
		model.compile(optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
		History = model.fit(Trainx, Trainy, epochs=100, batch_size=64, validation_data=(Vaildx, Vaildy))
		pre = model.evaluate(Testx, Testy, batch_size=64, verbose=2)
		print('test_loss:', pre[0], '- test_acc:', pre[1])
		plt.figure(figsize=(15, 5))
		plt.subplot(1, 2, 1)
		# df = pd.DataFrame.from_dict(History.history, orient="index")
		# df.to_csv(os.path.join(data_csv_path,Model_name+"V2.csv"))
		plt.plot(History.history['accuracy'])
		plt.plot(History.history['val_accuracy'])
		plt.title('CNN accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.subplot(1, 2, 2)
		plt.plot(History.history['loss'])
		plt.plot(History.history['val_loss'])
		plt.title('CNN loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		plt.show()

	elif Model_name == "VGGCONV1":
		model = VGGCONV1(46,586,24)
		Trainx=Trainx.reshape(Trainx.shape[0],46,586)
		Vaildx=Vaildx.reshape(Vaildx.shape[0],46,586)
		Testx = Testx.reshape(Testx.shape[0],46,586)
		print(Trainx.shape)
		print(Testx.shape)
		print(Vaildx.shape)
		print(Trainy.shape)
		print(Vaildy.shape)
		print(Testy.shape)
		model.summary()
		model.compile(optimizer=Adam(lr=0.00025, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
		History = model.fit(Trainx, Trainy, epochs=200, batch_size=64, validation_data=(Vaildx, Vaildy))
		pre = model.evaluate(Testx, Testy, batch_size=64, verbose=2)
		print('test_loss:', pre[0], '- test_acc:', pre[1])
		plt.figure(figsize=(15, 5))
		plt.subplot(1, 2, 1)
		# df = pd.DataFrame.from_dict(History.history, orient="index")
		# df.to_csv(os.path.join(data_csv_path,Model_name+"V2.csv"))
		plt.plot(History.history['accuracy'])
		plt.plot(History.history['val_accuracy'])
		plt.title('CNN accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.subplot(1, 2, 2)
		plt.plot(History.history['loss'])
		plt.plot(History.history['val_loss'])
		plt.title('CNN loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		plt.show()
	elif Model_name=="AllConv":
		model = AllConv(166, 586, 1, 7)
		print(Trainx.shape)
		print(Testx.shape)
		print(Vaildx.shape)
		print(Trainy.shape)
		print(Vaildy.shape)
		print(Testy.shape)
		model.summary()
		model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
		History = model.fit(Trainx, Trainy, epochs=30, batch_size=64, validation_data=(Vaildx, Vaildy))
		pre = model.evaluate(Testx, Testy, batch_size=64, verbose=2)
		print('test_loss:', pre[0], '- test_acc:', pre[1])
		plt.figure(figsize=(15, 5))
		plt.subplot(1, 2, 1)
		plt.plot(History.history['accuracy'])
		plt.plot(History.history['val_accuracy'])
		plt.title('CNN accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')

		plt.subplot(1, 2, 2)
		plt.plot(History.history['loss'])
		plt.plot(History.history['val_loss'])
		plt.title('CNN loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		plt.show()
	elif Model_name=="Inception":
		model = InceptionV1(166, 586, 1,12)
		print(Trainx.shape)
		print(Testx.shape)
		print(Vaildx.shape)
		print(Trainy.shape)
		print(Vaildy.shape)
		print(Testy.shape)
		model.summary()
		model.compile(optimizer=Adam(lr=0.00025, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
		History = model.fit(Trainx, Trainy, epochs=40, batch_size=64, validation_data=(Vaildx, Vaildy))
		pre = model.evaluate(Testx, Testy, batch_size=64, verbose=2)
		print('test_loss:', pre[0], '- test_acc:', pre[1])
		plt.figure(figsize=(15, 5))
		plt.subplot(1, 2, 1)
		plt.plot(History.history['accuracy'])
		plt.plot(History.history['val_accuracy'])
		plt.title('CNN accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')

		plt.subplot(1, 2, 2)
		plt.plot(History.history['loss'])
		plt.plot(History.history['val_loss'])
		plt.title('CNN loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		plt.show()
	elif Model_name == "ResNet":
		model = ResNet(166, 586, 1, 24)
		print(Trainx.shape)
		print(Testx.shape)
		print(Vaildx.shape)
		print(Trainy.shape)
		print(Vaildy.shape)
		print(Testy.shape)
		model.summary()
		model.compile(optimizer=Adam(lr=0.00015, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
		History = model.fit(Trainx, Trainy, epochs=60, batch_size=64)
		# , validation_data=(Vaildx, Vaildy)
		# pre = model.evaluate(Testx, Testy, batch_size=64, verbose=2)
		# print('test_loss:', pre[0], '- test_acc:', pre[1])
		# df = pd.DataFrame.from_dict(History.history, orient="index")
		# df.to_csv(os.path.join(data_csv_path,Model_name+".csv"))
		# plt.figure(figsize=(15, 5))
		# plt.subplot(1, 2, 1)
		# plt.plot(History.history['accuracy'])
		# plt.plot(History.history['val_accuracy'])
		# plt.title('CNN accuracy')
		# plt.ylabel('accuracy')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'test'], loc='upper left')
		model.save("ResNet24Tone.h5")

		# plt.subplot(1, 2, 2)
		# plt.plot(History.history['loss'])
		# plt.plot(History.history['val_loss'])
		# plt.title('CNN loss')
		# plt.ylabel('loss')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'test'], loc='upper left')
		# plt.show()
		# plt.show()

	else:
		print("Model Name is not Specify")





