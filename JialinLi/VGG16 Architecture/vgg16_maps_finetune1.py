import numpy as np
import os
import time
import keras
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection  import train_test_split

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print (x.shape)
x = np.expand_dims(x, axis=0)
print (x.shape)
x = preprocess_input(x)
print('Input image shape:', x.shape)

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/maps for classification of regions'
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
#		x = x/255
		print('Input image shape:', x.shape)
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
        model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
        model.summary()

        last_layer = model.get_layer('block5_pool').output
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
        filename='Experiment results_pretrained1_SGD'+'.txt'
        file = open(filename,'a')
        file.write(str1) 
        file.write(str2)
        file.write(str3)
        file.close()