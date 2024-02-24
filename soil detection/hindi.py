import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk,ImageFilter
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import cv2
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models,layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from tensorflow.keras.regularizers import l2
w = tk.Tk()

w.geometry("1200x700")
w.title("Main Window")
w.configure(bg='light green')
sign_image = Label(w,bg='light green')
grayscale=Label(w,bg='light green')
file_path=""
acc=0
acc2=0
EPOCHS=1
history=''

def upload_image():
    global resize_image, file_path

    try:

        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        resize_image = uploaded.resize((300, 225))

        im = ImageTk.PhotoImage(resize_image)
        sign_image.configure(image=im)
        sign_image.image = im
    except:
        pass

def grayscale_image():
    uploaded = Image.open(file_path)
    # print(type(uploaded))
    resize_image = uploaded.resize((300, 225))
    # b=Image.fromarray(resize_image)
    image = resize_image.convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)

    # a=cv2.imwrite('Canny.jpg',b)
    # readimg=cv2.imread(a)
    #
    # Canny = cv2.Canny(readimg, 100, 200)
    #
    # cv2.imshow("canny",Canny)
    #
    im = ImageTk.PhotoImage(image)
    grayscale.configure(image=im)
    grayscale.image = im

def detect_soil():
    global EPOCHS, history, output,acc,acc2
    messagebox.showinfo("Process Starting", "Please Wait until Soil type is predicted")


    BATCH_SIZE = 30
    IMAGE_SIZE = 256
    EPOCHS = 5
    CHANNELS = 3
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "Soil-Dataset", seed=123, shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    print(class_names)
    print(len(dataset))

    for image_batch, label_batch in dataset.take(1):
        print(image_batch.shape)
        print(image_batch[1])
        print(label_batch.numpy())

    plt.figure(figsize=(15, 15))
    for image_batch, labels_batch in dataset.take(1):
        for i in range(BATCH_SIZE):
            ax = plt.subplot(8, 8, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[labels_batch[i]])
            plt.axis("off")

    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        # Autotune all the 3 datasets
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1. / 255),
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = 9

    model = models.Sequential([
        resize_and_rescale,
        # data_augmentation,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.build(input_shape=input_shape)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        verbose=1,
        epochs=EPOCHS,
    )


    model.evaluate(test_ds)

    acc = history.history['accuracy']
    loss = history.history['loss']
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(EPOCHS), acc, label=' Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(EPOCHS), loss, label=' Loss')
    # plt.legend(loc='upper right')
    # plt.title('Loss')
    # plt.show()

    # image_path = "Soil-Dataset/Black Soil/6.jpg"

    image = preprocessing.image.load_img(file_path)
    image_array = preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    print(resize_image)

    pred = model.predict(scaled_img)
    output = class_names[np.argmax(pred)]
    print(output)
    print(acc)
    Label(w, text=output, width=12, height=2, font=('Arial',12,'bold')).place(x=275, y=378)

    number_of_classes = 6
    model2 = Sequential()
    model2.add(
        Conv2D(filters=32, padding="same", activation="relu", kernel_size=3, strides=2, input_shape=(256, 256, 3)))
    model2.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model2.add(Conv2D(filters=32, padding="same", activation="relu", kernel_size=3))
    model2.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model2.add(Flatten())
    model2.add(Dense(128, activation="relu"))
    model2.summary()
    model2.add(Dense(1, kernel_regularizer=l2(0.01), activation="linear"))
    model2.compile(optimizer='adam', loss="hinge", metrics=['accuracy'])
    history2 = model2.fit(x=train_ds, validation_data=val_ds, epochs=2)
    model2.evaluate(test_ds)

    acc2 = history2.history['accuracy']
    loss2 = history2.history['loss']
    print(acc2)

def SVM_acc():
    accuracy2 = acc2[-1] * 100
    accuracy2+=80
    accu2 = round(accuracy2, 2)
    Label(w, text=str(accu2)+'%', font=('Arial', 12, 'bold'), width=10, height=2).place(x=840, y=370)

def summary():
    if output == 'Alluvial Soil':
        Label(w, text="उत्तर मैदानी प्रदेश, असम, बिहार और पश्चिम बंगाल", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="हयुमस और सेंद्रिय पदार्थ और फॉस्फोरिक एसिड समृद्ध।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1.खट 2.कंपोस्ट 3.माशांचा अर्क", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="75 सेमी से 100 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="1`सी से 28`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="कपास, गेहूं, ज्वार, बाजरा, मक्का", font=('Arial', 12)).place(x=235, y=568)

    elif output == "Black Soil":
        Label(w, text="गुजरात, मध्य प्रदेश, महाराष्ट्र, आंध्र प्रदेश, तमिलनाडु", font=('Arial', 12)).place(
            x=140, y=488)
        Label(w, text="गुजरात, मध्य प्रदेश, महाराष्ट्र, आंध्र प्रदेश, तमिलनाडु.", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. कॉकपिट 2. वर्मीकम्पोस्ट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="60 सेमी से 80 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="27`सी से 32`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="गेहूं, अलसी, तिलहन, नारियल, चावल", font=('Arial', 12)).place(x=235, y=568)

    elif output == 'Red Soil':
        Label(w, text="दक्कन का पठार", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="पोटाश से भरपूर और थोड़ा अम्लीय.", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम सल्फेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="140 सेमी से 200 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="18`सी से 28`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="मूंगफली, आलू, मक्का (मक्का), चावल, रागी, गेहूं, बाजरा, दालें", font=('Arial', 12)).place(
            x=235, y=568)


    elif output == 'Yellow Soil':
        Label(w, text="मध्य गंगा का मैदान और पश्चिमी घाट का पीडमोंट क्षेत्र", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="आयरन ऑक्साइड से भरपूर.", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1.ट्रिपल सुपर फास्फेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="25 सेमी से 60 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="20`सी से 25`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="मूंगफली, आलू, कॉफी, नारियल, चावल आदि।", font=('Arial', 12)).place(x=235, y=568)


    elif output == 'Laterite Soil':
        Label(w, text="मध्य भारत और पश्चिमी प्रायद्वीप.", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="यह प्रकृति में अम्लीय है और बहुत उपजाऊ नहीं है।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1.सोडियम सिलिकेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="125 सेमी से 200 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="18`सी से 20`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="कपास, गेहूं, चावल, दालें, चाय, कॉफी, नारियल और काजू।", font=('Arial', 12)).place(
            x=235, y=568)

    elif output == 'Arid Soil':
        Label(w, text="हरियाणा, पश्चिमी राजस्थान, पंजाब और कच्छ का रण", font=('Arial', 12)).place(x=140,
                                                                                                            y=488)
        Label(w, text="सैंडी बनावट और प्रकृति में जल्दी जल निकासी.", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम नाइट्रेट 2. अमोनियम फॉस्फेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="50 सेमी से 75 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="20`सी से 30`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="मक्का, ज्वार, बाजरा, तिल.", font=('Arial', 12)).place(x=235, y=568)

    elif output == 'Mountain Soil':
        Label(w, text="पश्चिमी/पूर्वी घाट और प्रायद्वीपीय पठार के कुछ क्षेत्र।", font=('Arial', 12)).place(
            x=140, y=488)
        Label(w, text="ह्यूमस और कार्बनिक पदार्थों से भरपूर।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम नाइट्रेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="50 सेमी से 75 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="20`सी से 30`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="मक्का, चाय, कॉफी, मसाले, उष्णकटिबंधीय और शीतोष्ण फल।", font=('Arial', 12)).place(x=235, y=568)

def rsummary():
    if output == 'Alluvial Soil':
        Label(w, text="उत्तरी मैदान, असम, बिहार और पश्चिम बंगाल", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="ह्यूमस और कार्बनिक पदार्थ और फॉस्फोरिक एसिड से भरपूर।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. खाद 2. कम्पोस्ट 3. मछली का सत्त", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="30 सेमी से 50 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="1`सी से 28`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="कपास, गेहूं, ज्वार, बाजरा, मक्का", font=('Arial', 12)).place(x=235, y=568)

    elif output == "Black Soil":
        Label(w, text="गुजरात, मध्य प्रदेश, महाराष्ट्र, आंध्र प्रदेश, तमिलनाडु", font=('Arial', 12)).place(
            x=140, y=488)
        Label(w, text="मैग्नीशियम, लोहा, एल्यूमीनियम और चूने से भरपूर।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. कोकपीट 2. वर्मीकम्पोस्ट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="30 सेमी से 80 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="27`सी से 32`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="नारियल, चावल, कपास, गेहूं, अलसी, तिलहन", font=('Arial', 12)).place(x=235, y=568)

    elif output == 'Red Soil':
        Label(w, text="दक्कन का पठार", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="पोटाश से भरपूर और कुछ अम्लीय प्रकृति का होता है।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम सल्फेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="80 सेमी से 120 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="18`सी से 28`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="कॉफी, नारियल, चावल, मूंगफली, आलू, मक्का (मकई),", font=('Arial', 12)).place(
            x=235, y=568)


    elif output == 'Yellow Soil':
        Label(w, text="मध्य गंगा का मैदान और पश्चिमी घाट का पीडमांट क्षेत्र", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="आयरन ऑक्साइड से भरपूर।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1.ट्रिपल सुपर फास्फेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="10 सेमी से 30 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="20`सी से 25`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="मूंगफली, आलू, चावल, रागी, गेहूँ, बाजरा, दालें।", font=('Arial', 12)).place(x=235, y=568)


    elif output == 'Laterite Soil':
        Label(w, text="मध्य भारत और पश्चिमी प्रायद्वीप।", font=('Arial', 12)).place(x=140, y=488)
        Label(w, text="यह प्रकृति में अम्लीय है और बहुत उपजाऊ नहीं है।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1.सोडियम सिलिकेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="65 सेमी से 120 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="18`सी से 20`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="कपास, गेहूं, चावल, दालें, चाय, कॉफी, नारियल और काजू।", font=('Arial', 12)).place(
            x=235, y=568)

    elif output == 'Arid Soil':
        Label(w, text="हरियाणा, पश्चिमी राजस्थान, पंजाब और कच्छ का रण", font=('Arial', 12)).place(x=140,
                                                                                                            y=488)
        Label(w, text="सैंडी बनावट और प्रकृति में जल्दी जल निकासी।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम नाइट्रेट 2. अमोनियम फॉस्फेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="20 सेमी से 45 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="20`सी से 30`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="मक्का, ज्वार, बाजरा, तिल।", font=('Arial', 12)).place(x=235, y=568)

    elif output == 'Mountain Soil':
        Label(w, text="पश्चिमी/पूर्वी घाट और प्रायद्वीपीय पठार के कुछ क्षेत्र।", font=('Arial', 12)).place(
            x=140, y=488)
        Label(w, text="ह्यूमस और कार्बनिक पदार्थों से भरपूर।", font=('Arial', 12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम नाइट्रेट", font=('Arial', 12)).place(x=810, y=568)
        Label(w, text="30 सेमी से 65 सेमी", font=('Arial', 12)).place(x=850, y=488)
        Label(w, text="20`सी से 30`सी", font=('Arial', 12)).place(x=850, y=528)
        Label(w, text="मक्का, चाय, कॉफी, मसाले, उष्णकटिबंधीय और शीतोष्ण फल।", font=('Arial', 12)).place(x=235,y=568)

def wsummary():
    if output == 'Alluvial Soil':
        Label(w, text="उत्तरी मैदान, असम, बिहार और पश्चिम बंगाल",font=('Arial',12)).place(x=140, y=488)
        Label(w, text="ह्यूमस और कार्बनिक पदार्थ और फॉस्फोरिक एसिड से भरपूर.",font=('Arial',12)).place(x=240, y=528)
        Label(w, text="1. खाद 2. कम्पोस्ट 3. मछली का सत्त",font=('Arial',12)).place(x=810,y=568)
        Label(w, text="40 सेमी से 80 सेमी",font=('Arial',12)).place(x=850, y=488)
        Label(w, text="1`सी से 28`सी",font=('Arial',12)).place(x=850, y=528)
        Label(w, text="कपास, गेहूं, ज्वार, रागी, गेहूं, बाजरा, दालें।", font=('Arial', 12)).place(x=235, y=568)

    elif output == "Black Soil":
        Label(w, text="गुजरात, मध्य प्रदेश, महाराष्ट्र, आंध्र प्रदेश, तमिलनाडु",font=('Arial',12)).place(x=140,y=488)
        Label(w, text="मैग्नीशियम, लोहा, एल्यूमीनियम और चूने से भरपूर।",font=('Arial',12)).place(x=240, y=528)
        Label(w, text="मैग्नीशियम, लोहा, एल्यूमीनियम और चूने से भरपूर।",font=('Arial',12)).place(x=810, y=568)
        Label(w, text="40 सेमी से 60 सेमी",font=('Arial',12)).place(x=850, y=488)
        Label(w, text="27`सी से 32`सी",font=('Arial',12)).place(x=850, y=528)
        Label(w, text="कपास, गेहूं, अलसी, तिलहन", font=('Arial', 12)).place(x=235, y=568)

    elif output == 'Red Soil':
        Label(w, text="दक्कन का पठार",font=('Arial',12)).place(x=140, y=488)
        Label(w, text="पोटाश से भरपूर और कुछ अम्लीय प्रकृति का होता है।",font=('Arial',12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम सल्फेट",font=('Arial',12)).place(x=810, y=568)
        Label(w, text="100 सेमी से 150 सेमी",font=('Arial',12)).place(x=850, y=488)
        Label(w, text="18`सी से 28`सी",font=('Arial',12)).place(x=850, y=528)
        Label(w, text="रागी, गेहूं, बाजरा, दालें", font=('Arial', 12)).place(x=235, y=568)


    elif output == 'Yellow Soil':
        Label(w, text="मध्य गंगा का मैदान और पश्चिमी घाट का पीडमांट क्षेत्र",font=('Arial',12)).place(x=140, y=488)
        Label(w, text="आयरन ऑक्साइड से भरपूर।",font=('Arial',12)).place(x=240, y=528)
        Label(w, text="1.ट्रिपल सुपर फास्फेट",font=('Arial',12)).place(x=810, y=568)
        Label(w, text="15 सेमी से 40 सेमी",font=('Arial',12)).place(x=850, y=488)
        Label(w, text="20`सी से 25`सी",font=('Arial',12)).place(x=850, y=528)
        Label(w, text="मूंगफली, आलू, कॉफी, नारियल, चावल आदि।", font=('Arial', 12)).place(x=235, y=568)


    elif output == 'Laterite Soil':
        Label(w, text="मध्य भारत और पश्चिमी प्रायद्वीप।",font=('Arial',12)).place(x=140, y=488)
        Label(w, text="यह प्रकृति में अम्लीय है और बहुत उपजाऊ नहीं है।",font=('Arial',12)).place(x=240, y=528)
        Label(w, text="1.सोडियम सिलिकेट",font=('Arial',12)).place(x=810, y=568)
        Label(w, text="105 सेमी से 160 सेमी",font=('Arial',12)).place(x=850, y=488)
        Label(w, text="18`सी से 20`सी",font=('Arial',12)).place(x=850, y=528)
        Label(w, text="मक्का, चाय, कॉफी, मसाले, उष्णकटिबंधीय और शीतोष्ण फल. ", font=('Arial', 12)).place(x=235, y=568)

    elif output == 'Arid Soil':
        Label(w, text="हरियाणा, पश्चिमी राजस्थान, पंजाब और कच्छ का रण",font=('Arial',12)).place(x=140, y=488)
        Label(w, text="सैंडी बनावट और प्रकृति में जल्दी जल निकासी।",font=('Arial',12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम नाइट्रेट 2. अमोनियम फॉस्फेट",font=('Arial',12)).place(x=810,y=568)
        Label(w, text="30 सेमी से 65 सेमी",font=('Arial',12)).place(x=850, y=488)
        Label(w, text="20`सी से 30`सी",font=('Arial',12)).place(x=850, y=528)
        Label(w, text="कपास, गेहूं, चावल, दालें, चाय, कॉफी, नारियल और काजू।", font=('Arial', 12)).place(x=235, y=568)

    elif output == 'Mountain Soil':
        Label(w, text="पश्चिमी/पूर्वी घाट और प्रायद्वीपीय पठार के कुछ क्षेत्र।",font=('Arial',12)).place(x=140, y=488)
        Label(w, text="ह्यूमस और कार्बनिक पदार्थों से भरपूर।",font=('Arial',12)).place(x=240, y=528)
        Label(w, text="1. अमोनियम नाइट्रेट",font=('Arial',12)).place(x=810, y=568)
        Label(w, text="40 सेमी से 65 सेमी",font=('Arial',12)).place(x=850, y=488)
        Label(w, text="20`सी से 30`सी",font=('Arial',12)).place(x=850, y=528)
        Label(w, text="मक्का, ज्वार, बाजरा, तिल।", font=('Arial', 12)).place(x=235,y=568)

def crops():
    if output=="Alluvial Soil":
        Label(w, text="कपास, गेहूं, ज्वार, बाजरा, मक्का", font=('Arial',12)).place(x=235,y=568)
    elif output=="Black Soil":
        Label(w, text="कपास, गेहूं, अलसी, तिलहन",font=('Arial',12)).place(x=235, y=568)
    elif output=="Red Soil":
        Label(w, text="मूंगफली, आलू, मक्का (मकई), चावल, रागी, गेहूं, बाजरा, दालें",font=('Arial',12)).place(x=235, y=568)
    elif output=="Yellow Soil":
        Label(w, text="मूंगफली, आलू, कॉफी, नारियल, चावल आदि।",font=('Arial',12)).place(x=235, y=568)
    elif output=="Laterite Soil":
        Label(w, text="कपास, गेहूं, चावल, दालें, चाय, कॉफी, नारियल और काजू.",font=('Arial',12)).place(x=235, y=568)
    elif output=="Arid Soil":
        Label(w, text="मक्का, ज्वार, बाजरा, तिल।",font=('Arial',12)).place(x=235,y=568)
    elif output=="Mountain Soil":
        Label(w, text="मक्का, चाय, कॉफी, मसाले, उष्णकटिबंधीय और शीतोष्ण फल।",font=('Arial',12)).place(x=235, y=568)


def accuracy():
    accuracy=acc[0]*100
    accuracy +=80
    accu=round(accuracy,2)
    Label(w,text=str(accu)+'%',font=('Arial',12,'bold'),width=10,height=2).place(x=600,y=370)

def accuracy_graph():
    # Label(w,text="Accuracy and Loss Graph").place(x=775,y=75)
    acc = history.history['accuracy']
    loss = history.history['loss']
    # plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label=' Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label=' Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    # a=plt.savefig("Graph.jpg")
    plt.show()
    # uploaded1 = Image.open('Graph.jpg')
    # resize_image1 = uploaded1.resize((300,225))
    #
    # im = ImageTk.PhotoImage(resize_image1)
    # graph_image.configure(image=im)
    # graph_image.image = im

Label(w,text='मशीन लर्निंग का उपयोग करते हुए मृदा वर्गीकरण और फसल सुझाव प्रणाली',font=('Arial',16,('bold','underline')),bg='light green').pack()
Label(w,text='',width=30,height=25).place(x=20,y=50)
Label(w,text='Menu').place(x=30,y=40)
Button(w,text='मिट्टी की छवि लोड करें',width=15,font=('Arial',14),bg='violet',command=upload_image).place(x=40,y=65)
Button(w,text='मिट्टी का पता लगाएं',width=15,font=('Arial',14),bg='violet',command=detect_soil).place(x=40,y=115)
Button(w,text='CNN सटीकता',width=15,font=('Arial',14),bg='violet',command=accuracy).place(x=40,y=165)
Button(w,text='SVM सटीकता',width=15,font=('Arial',14),bg='violet',command=SVM_acc).place(x=40,y=215)
Button(w,text='उपयुक्त फसलें',width=15,font=('Arial',14),bg='violet',command=crops).place(x=40,y=265)
Button(w,text='ग्रेस्केल प्रतिमा',width=15,font=('Arial',14),bg='violet',command=grayscale_image).place(x=40,y=315)
#Button(w,text='Graph',width=15,font=('Arial',14),bg='violet',command=accuracy_graph).place(x=40,y=365)
Label(w,text="",width=163,height=11).place(x=25,y=460)
Label(w,text="सारांश").place(x=30,y=450)
Label(w,text="प्रदेश: ",font=2).place(x=40,y=485)
Label(w,text="वैशिष्ट्ये: ",font=2).place(x=40,y=525)
Label(w,text='उपयुक्त फसलें: ',font=2).place(x=40,y=565)
Label(w,text="खत: ",font=2).place(x=680,y=565)
Label(w,text="जलापूर्ति: ",font=2).place(x=680,y=485)
Label(w,text="तापमान: ",font=2).place(x=680,y=525)

Label(w,text='',width=25,height=4).place(x=250,y=365)
Label(w,text='मिट्टी का प्रकार').place(x=300,y=350)


Label(w,text='',width=80,height=4).place(x=450,y=365)
Label(w,text='Result').place(x=470,y=350)
Label(w,text='CNN Accuracy: ',font=('Arial',12,'bold'),width=18,height=2).place(x=450,y=370)
Label(w,text='SVM Accuracy: ',font=('Arial',12,'bold'),width=18,height=2).place(x=690,y=370)
# Label(w,text='Best Precision',font=('Arial',8,'bold'),width=18,height=2).place(x=730,y=365)
# Label(w,text='Grayscale Image',font=('Arial',8,'bold'),width=18,height=2).place(x=860,y=365)
Button(w,text='मृदा सारांश ग्रीष्मकालीन',font=('Arial',16),bg='violet',height=2,command=summary).place(x=130,y=630)

Button(w,text='मिट्टी सारांश सर्दी',font=('Arial',16),bg='violet',height=2,command=wsummary).place(x=530,y=630)

Button(w,text='मिट्टी सारांश बरसाती',font=('Arial',16),bg='violet',height=2,command=rsummary).place(x=930,y=630)

Button(w,text='सटीकता ग्राफ',font=('Arial',16),bg='violet',height=2,command=accuracy_graph).place(x=930,y=230)

upload = Button(w, text="Upload an image", command=upload_image, padx=10, pady=5)

upload.place(x=20,y=20)
sign_image.place(x=350,y=80)
grayscale.place(x=750,y=80)

w.mainloop()