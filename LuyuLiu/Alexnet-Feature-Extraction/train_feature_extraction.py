import pickle
import time
# import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from alexnet import AlexNet


nb_classes = 43
EPOCHS = 10
BATCH_SIZE = 128

# TODO: Load traffic signs data.
training_file = 'D:\\Luyu\\ML_map\\ML-Final-Project\\LuyuLiu\\Alexnet-Feature-Extraction\\train.p'
with open(training_file, mode='rb') as f:
    data = pickle.load(f)





# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(
    data['features'], data['labels'], test_size=0.33, random_state=42)

print(X_train.shape, X_val.shape, y_train.shape)

path='C:\\Users\\li.7957\\Desktop\\Dropbox\\Dissertation Materials\\Images for training\\maps for classification of regions\\'
path_source1=path+'world maps\\'
path_source2=path+'China maps\\'
path_source3=path+'South Korea maps\\'
path_source4=path+'US maps\\'

# num_maps_class
width=120
height=100
num_pixels=width*height
input_size=width*height*3
input_shape=(width, height, 3)

num_classes = 4

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

num_list=[240,280,320,360,400] # total number of images used
for num in num_list:

    num_total=num
    num_test=40
    num_train=num_total-num_test
    num_map_region=int(num_total/4)

    str1="train size:"+str(num_train)+' test size:'+str(num_test)+'\n'
    print(str1)
    data_pair=[]

    # Get the image data and store data into X_batches and y_batches

    for i in range(num_map_region):
        name_source='map'+str(i+1)+'.jpg'
        img = Image.open(path_source1+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        # print(len(pixel_values))
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='china_map'+str(i+1)+'.jpg'
        img = Image.open(path_source2+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='south_korea_map'+str(i+1)+'.jpg'
        img = Image.open(path_source3+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        # print(len(pixel_values))
        data_pair.append(pixel_values)

    for i in range(num_map_region):
        name_source='us_map'+str(i+1)+'.jpg'
        img = Image.open(path_source4+name_source)
        img = img.resize((width, height), Image.ANTIALIAS)
        pixel_values=list(img.getdata())
        data_pair.append(pixel_values)

    data_pair_3=[]
    for i in range(num_total):
        pixel_value_list=[]
        for j in range(num_pixels):
            # print("j:",j)
            pixels=data_pair[i][j]
            try:
                pixel_value_list.append(pixels[0])
                pixel_value_list.append(pixels[1])
                pixel_value_list.append(pixels[2])
            except:
                print("i:",i)
                break
        if i<num_map_region:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[0]+[i])
        elif i>=num_map_region and i < num_map_region*2:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[1]+[i])
        elif i>=num_map_region*2 and i < num_map_region*3:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[2]+[i])
        elif i>=num_map_region*3 and i < num_map_region*4:
            # print(len(pixel_value_list))
            data_pair_3.append(pixel_value_list+[3]+[i])

    len_x=len(data_pair_3[0])-2
    inx_y=len_x+1
    inx_image=inx_y+1
    # Shuffle data_pair as input of Neural Network
    # random.seed(42)
    test_loss_list=[]
    test_acc_list=[]
    train_time_list=[]
    test_time_list=[]

    for inx in range(10):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                        activation='relu',
                        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(lr=0.01),
                    metrics=['accuracy'])
        X_batches=[]
        y_batches=[]
        print("sets of experiments",inx)
        random.shuffle(data_pair_3)
        data_pair=np.array(data_pair_3)

        index_image_list=[]
        for i in range(num_total-num_test,num_total):
            index_image_list.append(data_pair_3[i][inx_image-1]+1)

        X_batches_255=[data_pair_3[i][0:len_x] for i in range(num_total)]  
    
        y_batches=[data_pair_3[i][len_x] for i in range(num_total)]
        # data get from last step is with the total value of pixel 255 

        for i in range(num_total):
            X_1img=[X_batches_255[i][j]/255.0 for j in range(len_x)]
            X_batches.append(X_1img)
        X_batches=np.array(X_batches)
        y_batches=np.array(y_batches)

        x_train=X_batches[0:num_train].reshape(num_train,input_size)
        x_test=X_batches[num_train:num_total].reshape(num_test,input_size)
        y_train=y_batches[0:num_train].reshape(num_train,1)
        y_test=y_batches[num_train:num_total].reshape(num_test,1)
        
        x_train = x_train.reshape(x_train.shape[0], width, height, 3)
        x_test = x_test.reshape(x_test.shape[0], width, height, 3)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)











# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels = tf.placeholder(tf.int32, None)
resized = tf.image.resize_images(x, [227, 227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1e-01))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(labels, nb_classes)
rate = 0.001

# keep_prob=tf.placeholder(tf.float32)

# For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

# preds = tf.arg_max(logits, 1) # Not use one_hot_y here
# accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_loss = 0
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        #loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, labels: batch_y, keep_prob: 1.0})
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={
                                  x: batch_x, labels: batch_y})
        total_loss += (loss * batch_x.shape[0])
        ltotal_accuracy += (accuracy * len(batch_x))

    return total_loss / num_examples, total_accuracy / num_examples


# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    #num_examples = X_train.shape[0]
    print("Training...")
    print()
    iter = 0

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            #sess.run(training_operation, feed_dict={x: batch_x, labels: batch_y, keep_prob: 0.5})
            sess.run(training_operation, feed_dict={
                     x: batch_x, labels: batch_y})

        validation_loss, validation_accuracy = evaluate(X_val, y_val)

        print("EPOCH {} ...".format(i+1))
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print(" ")

    saver.save(sess, './AlexNet')
    print("Model saved")
