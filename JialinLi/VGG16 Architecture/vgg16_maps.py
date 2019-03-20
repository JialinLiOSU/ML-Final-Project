import numpy as np
import os
import time
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

labels[0:100]=0
labels[100:200]=1
labels[200:300]=2
labels[300:400]=3

names = ['china','south_korea','us','world']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#########################################################################################
# Custom_vgg_model_1
#Training the classifier alone
batch_size=32 
epochs_list=[20,40,60,80,100]
for epochs in epochs_list:
	image_input = Input(shape=(224, 224, 3))

	model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
	model.summary()
	last_layer = model.get_layer('fc2').output
	#x= Flatten(name='flatten')(last_layer)
	out = Dense(num_classes, activation='softmax', name='output')(last_layer)
	custom_vgg_model = Model(image_input, out)
	custom_vgg_model.summary()

	for layer in custom_vgg_model.layers[:-1]:
		layer.trainable = False

	print(custom_vgg_model.layers[3].trainable)

	custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

	t=time.time()
	#	t = now()
	hist = custom_vgg_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
	print('Training time: %s' % (time.time()-t))
	(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

	print()

	str1="Batch_size: "+str(batch_size)+' epochs: '+str(epochs)+'\n'
	str2="[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100)+'\n'
		
	filename='Experiment results_pretrained'+'.txt'
	file = open(filename,'a')
	file.write(str1) 
	file.write(str2)
	file.close() 
