import numpy as np
import os
import time
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection  import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from keras.layers import Activation, Conv2D
from keras.layers import Input
from keras.layers import merge
from keras.models import Model
from keras.optimizers import SGD
from scipy.misc import imread
from scipy.misc import imresize
from keras.utils import plot_model 

def VGG_19(weights_path=None, input_tensor=None):
    img_input = input_tensor
# Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='last_layer')(x)


    return Model(img_input, x, name='vgg19')



base_location = "D:/Luyu/ML_map"
data_location = base_location + "/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
project_location = "D:/Luyu/ML_map/ML-Final-Project"
img_path = project_location+'/JialinLi/VGG16 Architecture/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print (x.shape)
x = np.expand_dims(x, axis=0)
print (x.shape)
x = preprocess_input(x)
print('Input image shape:', x.shape)

# Loading the training data
PATH = project_location + '/JialinLi/VGG16 Architecture/maps for classification of regions'
# Define data path
data_path = PATH
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
num_img_each=100

labels[0:num_img_each]=0
labels[num_img_each:num_img_each*2]=1
labels[num_img_each*2:num_img_each*3]=2
labels[num_img_each*3:num_img_each*4]=3

names = ['china','south_korea','us','world']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

####################################################################################################################

#Training the feature extraction also
batch_size=16
epochs_list=[20,40,60,80,100]
layer_inx_list=[-3,-4,-5,-6,-7,-8,-9,-10]
for layer_inx in layer_inx_list:
    for epochs in epochs_list:
        image_input = Input(shape=(224, 224, 3))
        model = VGG_19(weights_path=data_location, input_tensor=image_input)
        
        model.load_weights(data_location)
        plot_model(model, to_file=base_location + '/model.png')
        model.summary()

        last_layer = model.get_layer('last_layer').output
        x= Flatten(name='flatten')(last_layer)
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        out = Dense(num_classes, activation='softmax', name='output')(x)
        custom_vgg_model2 = Model(image_input, out)

        # freeze all the layers except the dense layers
        for layer in custom_vgg_model2.layers[:layer_inx]:
            layer.trainable = False

        custom_vgg_model2.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(lr=0.01),
                    metrics=['accuracy'])

        t=time.time()
        #	t = now()
        hist = custom_vgg_model2.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
        print('Training time: %s' % (time.time()-t))
        (loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

        print()
        str1="Batch_size: "+str(batch_size)+' epochs: '+str(epochs)+'\n'
        str2="[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100)+'\n'
        str3='Training time: ' + str (time.time()-t)+'\n'
        filename='results_pretrained1_SGD'+'.txt'
        file = open(filename,'a')
        file.write(str1) 
        file.write(str2)
        file.write(str3)
        file.close()