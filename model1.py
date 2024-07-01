import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def model_detector():
    # Define constants
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 32
    EPOCHS = 10

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/hemant soni/Desktop/Dataset2/Train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        'C:/Users/hemant soni/Desktop/Dataset2/Test',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    checkpoint_dir = 'model_weights'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    checkpoint_path = "model_weights/checkpoint.weights.h5"
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        monitor='val_accuracy',
                                        mode='max',
                                        save_best_only=True)

    # Train the model
    model.fit(train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=test_generator,
            validation_steps=test_generator.samples // BATCH_SIZE,
            callbacks=[checkpoint_callback])
    print(model.summary())
    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

    # Save the trained model
    model.save('model.h5')

    # Load the saved model
    # loaded_model = load_model('model.h5')

    # # Evaluate the loaded model
    # loss, accuracy = loaded_model.evaluate(test_generator)
    # print("Test Accuracy: {:.2f}%".format(accuracy * 100))

    return model
# model_detector()