import images
import pandas as pd
import numpy as np
import tensorflow as tf

import keras
from keras import models, layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.regularizers import l2
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

import asyncio
import websockets

from paho.mqtt import client as mqtt_client

# plot
from matplotlib import pyplot as plt
from matplotlib.image import imread

# beeld lader
import cv2

# extra
from datetime import datetime

#MQTT 

broker = ''
port = 1883
topic = "photos"
topic_sub = "photos"
client_id = ''

# static global vars
option = True
input_shape = 256
epochs = 10


def woven(dt):
    dt2 = dt[dt['Structure_and_Construction'] != 'Knit']
    return dt2


def knit(dt):
    dt2 = dt[dt['Structure_and_Construction'] != 'Woven']
    return dt2


def load(data_processed):
    data = tf.keras.utils.image_dataset_from_directory(data_processed)  # data genereren
    # %%
    class_names = data.class_names  # labels halen en bekijken
    print(class_names)

    return data


def split(i):
    len(i)  # lengte van de batch
    # splitsen 80% en 20%
    train_size = int(len(i) * .7)  # 70% om te trainen
    val_size = int(len(i) * .2)
    test_size = int(len(i) * .1) + 1  # 10% voor de test
    train = i.take(train_size)  # initialiseer train
    val = i.skip(train_size).take(val_size)
    test = i.skip(train_size + val_size).take(test_size)

    return train, val, test

# Our own model Agus
def model_one(t, v):
    m = models.Sequential()

    m.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape, input_shape, 3)))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Conv2D(32, (3, 3), activation='relu'))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Conv2D(64, (3, 3), activation='relu'))
    m.add(layers.MaxPooling2D((2, 2)))

    m.add(layers.Flatten())

    m.add(layers.Dense(128, activation='relu'))

    m.add(Dropout(0.5))

    m.add(layers.Dense(8, activation='softmax'))

    m.summary()

    m.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')

    h = m.fit(t, epochs=epochs, validation_data=v)

    return m, h

# Lenet by Vinesh
def model_two(t, v):
    m = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=5, strides=1,
                            activation='tanh', input_shape=(input_shape, input_shape, 3), padding='same'),  # C1
        keras.layers.AveragePooling2D(),  # S2
        keras.layers.Conv2D(32, kernel_size=5, strides=1,
                            activation='tanh', padding='valid'),  # C3
        keras.layers.AveragePooling2D(),  # S4
        keras.layers.Conv2D(64, kernel_size=5, strides=1,
                            activation='tanh', padding='valid'),  # C5
        keras.layers.Flatten(),
        keras.layers.Dense(84, activation='tanh'),  # F6
        keras.layers.Dense(8, activation='softmax')  # Output layer

    ])
    ln5m = m()

    h = ln5m.fit(t, epochs=epochs, validation_data=v)

    return ln5m, h

# Vgg19 with transferlearning Michal
def model_three(t, v):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(input_shape, input_shape, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the new model on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    h = model.fit(t, epochs=epochs, validation_data=v)
    return model, h

# Resnet by Tracy
def model_four(t, v):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_shape, input_shape, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the new model on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    h = model.fit(t, epochs=epochs, validation_data=v)
    return model, h

# Save trained model
def save_model(m):
    m.save("model.h5", include_optimizer=True)
    m.save_weights('model_weights.h5')

# Load trained model
def load_model():
    m = Model.load_model('model.h5')
    m.load_weights('model_weights.h5')
    return m


def accuracy_graph(h):
    plt.plot(h.history['accuracy'], label='accuracy')
    plt.plot(h.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def loss_graph(h):
    # fig = plt.figure()
    plt.plot(h.history['loss'], label='loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def evaluate_model(m, t):
    predicted_classes = m.evaluate(t)
    print(predicted_classes)

# Main code to process one image
def main(data):
        print(f"Data recieved as:  {data}!")

        dataset = pd.read_csv("P2_Fabric_data_csv.csv", encoding='ISO-8859-1', skipinitialspace=True)

        dataset.columns = [c.replace(' ', '_') for c in dataset.columns]

        if option == True:
            img = images.Data()
            img.sq_cut_img()

        try:
            model = load_model()
        except:
            if option == True:
                cnn_data = load(img.processed_data)

                # scale
                xMap = cnn_data.map(lambda x, y: (x / 255, y))  # de beeldwaarden downscalen naar tussen 0 en 1

                train, val, test = split(xMap)

                model, history = model_one(train, val)

                accuracy_graph(history)
                loss_graph(history)
                evaluate_model(model, train)

                save_model(model)

        if data is not None:

            x = image.img_to_array(data)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)
            print(prediction)
            return prediction
        else:
            return f"ERROR! Data recieved as:  {data}!"

async def handler(websocket, path):
 
    data = await websocket.recv()
    data = main(data)
    await websocket.send(data)

# Create MQTT
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Successfully connected to MQTT broker")
        else:
            print("Failed to connect, return code %d", rc)
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

# Subsicribe and create on message handler
def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        f = open('receive.jpg', 'wb')
        f.write(msg.payload)
        f.close()
        print ('image received')
        print ('Starting processing')
        main(cv2.imread('receive.jpg'))

    client.subscribe(topic_sub)
    client.on_message = on_message
    

if __name__ == "__main__":

    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()

    #start_server = websockets.serve(handler, "localhost", 8000)
    #asyncio.get_event_loop().run_until_complete(start_server)
    #asyncio.get_event_loop().run_forever()


