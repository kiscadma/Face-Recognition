import keras, glob, os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, random_rotation
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


def learn(target_name="George_W_Bush", target_count=530):
    all_data = []
    input_shape = (100, 100, 1)
    target_size = (100, 100) 
    color_mode = "grayscale"

    # grab all pictures of the target and add them with label 1
    pics = glob.glob('lfw/' + target_name + '/*.jpg')
    for pic in pics:
        all_data.append((img_to_array(load_img(pic, color_mode=color_mode, target_size=target_size)), 1))

    f = open('lfw-names.txt')
    for line in f:
        info = line.split()

        if info[0] is target_name:  # let's not add the target twice
            continue

        # add all photos for the non-target person with the 0 label
        pics = glob.glob('lfw/' + info[0] + '/*.jpg')
        for pic in pics:
            all_data.append((img_to_array(load_img(pic, color_mode=color_mode, target_size=target_size)), 0))

        # make sure we have an even split between target and non-target
        # having too many of either will skew training data
        # we are aiming for half target and half other
        if len(all_data) > target_count * 2:
            break

    # shuffle the data
    np.random.shuffle(all_data)
    trn_data = []
    trn_lbls = []
    test_data = []
    test_lbls = []
    train_test_split = int(.75 * len(all_data))

    # add some of data to the training data sets
    for pic_arr, lbl in all_data[:train_test_split]:
        trn_data.append(pic_arr)
        trn_data.append(random_rotation(pic_arr, 15))  # add a rotated version of the picture
        trn_lbls.append(lbl)
        trn_lbls.append(lbl)

    # add the remaining data to the test data sets
    for pic_arr, lbl in all_data[train_test_split:]:
        test_data.append(pic_arr)
        test_lbls.append(lbl)

    trn_data = np.array(trn_data)
    trn_lbls = np.array(trn_lbls)
    test_data = np.array(test_data)
    test_lbls = np.array(test_lbls)

    # construct the model
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model = keras.Sequential()
    model.add(Convolution2D(20, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(Convolution2D(40, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(60, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(80, kernel_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.74))
    model.add(Dense(1, activation='sigmoid'))  # Output Layer. 1==target, 0==not target
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    print("{}'s in the train dataset: {}/{}".format(target_name, np.sum(trn_lbls), len(trn_lbls)))
    model.fit(trn_data, trn_lbls, epochs=6)

    return model, test_data, test_lbls


if __name__ == "__main__":   
    cnn, data, labels = learn()

    # Evaluate model on test data
    print("\nFINAL EVALUATION RESULT...\n")
    print("Pictures of the target in the test dataset: {}/{}".format(np.sum(labels), len(labels)))
    acc = cnn.evaluate(x=data, y=labels)[1]

