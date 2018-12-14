from keras import utils as np_utils
import tensorflow as tf
import warnings
import os
import cv2
import keras.backend.tensorflow_backend as KTF
import numpy as np
import keras
from skimage import io, transform
from tflearn.layers.conv import global_avg_pool
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras import layers

# 進行配置，使用80%的GPU
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#session = tf.Session(config=config)

def main():
    Source = os.getcwd()

    MainFolder_Path = os.path.join(Source,'CroppedYale')

    Subfolder = os.listdir(MainFolder_Path)

    Training = []
    Testing = []
    recordTrain = []                   # recordTrain: An matrix to record train's index
    recordTest = []                    # recordTest: An matrix to record test's index

    for i in range(0, len(Subfolder)):
        SubFolder_Path = os.path.join(MainFolder_Path, Subfolder[i])
        cnt = 0
        for file in os.listdir(SubFolder_Path):
            if file.endswith(".pgm"):
                cnt += 1
                allImage = os.path.join(SubFolder_Path, file)

                if(cnt < 36):
                    Image = cv2.imread(allImage)
                    Items = cv2.resize(Image, (224, 224))/255
                    Training.append(Items)                     # Amplify the training matrix as a new image inserts
                    recordTrain.append(i)                       # Record the position
                else:
                    Image = cv2.imread(allImage)
                    Items = cv2.resize(Image, (224, 224))/255
                    Testing.append(Items)                      # Amplify the training matrix as a new image inserts
                    recordTest.append(i)                        # Record the position

    Training = np.array(Training, dtype = np.float64)
    Testing = np.array(Testing, dtype = np.float64)

    ans_Train = np_utils.to_categorical(recordTrain, num_classes=38)
    ans_Test = np_utils.to_categorical(recordTest, num_classes=38)

    model = VGG16(input_shape=[224, 224, 3])
    sgd = keras.optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights("model.h5", by_name=True) # Call Pre-train weights

    model.fit(Training, ans_Train, epochs=4, batch_size=1)

    score = model.evaluate(Testing, ans_Test, batch_size=1)
    print('\ntest loss: ', score[0])
    print('\ntest accuracy: ',score[1])

# 設置session
#KTF.set_session(session)

def VGG16(input_tensor=None, input_shape=None):

    img_input = Input(shape=input_shape)

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
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)

    # Only one Fully Connected
    x = Dense(4096, activation='relu', name='fca')(x)
    #x = Dense(4096, activation='relu', name='fcb')(x)
    x = Dense(38, activation='softmax', name='Classification')(x)


    inputs = img_input
    # Create model
    model = Model(inputs, x, name='vgg16')

    return model

if __name__ == "__main__":
    main()

#get_ipython().system('jupyter nbconvert --to script vgg16.ipynb')
