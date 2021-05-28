from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#initialize CNN
classifier=Sequential()

#convolutional layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(128,128,1),activation='relu',))
#classifier.add(Convolution2D(32,3,3,input_shape=(1,64,64),activation="relu"))

#pooling
classifier.add(MaxPooling2D(pool_size=(2,2),))

#flattening
classifier.add(Flatten())

#making ANN 
classifier.add(Dense(128,activation="relu"))
classifier.add(Dense(128,activation="relu"))
classifier.add(Dense(128,activation="relu"))

#outputlayer
classifier.add(Dense(46,activation="softmax"))

#compiling the CNN
classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])


####fitting images to CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )
test_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(
        'DevanagariHandwrittenCharacterDataset/Train',
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        color_mode="grayscale"
        )
test_set = test_datagen.flow_from_directory(
        'DevanagariHandwrittenCharacterDataset/Test',
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        color_mode="grayscale"
        )

classifier.fit(
        train_set,
        epochs=25,
        steps_per_epoch=78200//32,
        validation_data=test_set,
        validation_steps=13800//32
        )
fname="weight_cnn.hd.hdf5"
classifier.save_weights(fname)

