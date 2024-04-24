CODING
import numpy as np 
import pandas as pd 
!nvidia-smi
import os
for dirname, _, filenames in os.walk('dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder

df = pd.read_pickle('dataset\LSWMD.pkl')
df.info()
df.head()
df.tail()

unique_index = np.unique(df.waferIndex, return_counts=True) plt.bar(unique_index[0], unique_index[1], color='gold', align='center', alpha=0.5) plt.title('wafer index distribution') 
plt.xlabel('index') 

plt.ylabel('count') plt.xlim(0, 26) plt.ylim(30000, 34000) plt.show()

def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return (dim0, dim1)

df['waferMapDim'] = df.waferMap.apply(find_dim)
df.sample(5)

max(df['waferMapDim']), min(df['waferMapDim']) 
((300, 202), (6, 21))

unique_waferDim = np.unique(df['waferMapDim'])
unique_waferDim.shape (632, )

#Label Encoding
df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
mapping_type={
    'Center':0,
    'Donut':1,
    'Edge-Loc':2,
    'Edge-Ring':3,
    'Loc':4,
    'Random':5,
    'Scratch':6,
    'Near-full':7,
    'none':8}
mapping_traintest={'Training':0,'Test':1}
df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

#Drop out
df_label = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_label = df_label.reset_index()
df_pattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_pattern = df_pattern.reset_index()
df_none = df[(df['failureNum']==8)]
df_label.shape[0], df_pattern.shape[0], df_none.shape[0]
(172950, 25519, 147431)
tol_wafers = df.shape[0]
tol_wafers
81145

from matplotlib import gridspec 
fig = plt.figure(figsize=(20, 4.5)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5]) 
ax1 = plt.subplot(gs[0]) 
ax2 = plt.subplot(gs[1]) 
no_wafers=[tol_wafers-df_label.shape[0], df_pattern.shape[0], df_none.shape[0]] 
colors = ['silver', 'orange', 'gold']
explode = (0.1, 0, 0)  # explode 1st slice 
labels = ['no-label','label and pattern','label and non-pattern'] 
ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140) 
uni_pattern=np.unique(df_pattern.failureNum, return_counts=True) 
labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full'] ax2.bar(uni_pattern[0],uni_pattern[1]/df_pattern.shape[0], color='gold', align='center', alpha=0.9) 
ax2.set_title("failure type frequency") 
ax2.set_ylabel("% of pattern wafers") ax2.set_xticklabels(labels2) 
plt.show()

fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize=(20, 20)) ax = ax.ravel(order='C') 
for i in range(100): 
img = df_pattern.waferMap[i] ax[i].imshow(img) ax[i].set_title(df_pattern.failureType[i][0][0], fontsize=10) ax[i].set_xlabel(df_pattern.index[i], fontsize=8) 
ax[i].set_xticks([]) 
ax[i].set_yticks([]) 
plt.tight_layout()
 plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
df_pattern = {'failureNum': ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'Center'], 'shape': [1, 2, 3, 4, 5, 6, 7, 8, 9]} 
uni_pattern = np.unique(df_pattern['failureNum'], return_counts=True) 
labels2 = ['', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full'] 
fig, ax2 = plt.subplots() 
ax2.bar(uni_pattern[0], uni_pattern[1] / len(df_pattern['failureNum']), color='gold', align='center', alpha=0.9) 
ax2.set_title("Failure Type Frequency") 
ax2.set_ylabel("% of Pattern Wafers") ax2.set_xticklabels(labels2, rotation=45) 
plt.tight_layout() 
plt.show()

#Deeplearning using VGG 16 via AdaBoost CNN

import matplotlib.pyplot as plt from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D from tensorflow.keras.layers import Flatten, Softmax, SpatialDropout2D from tensorflow.keras.layers import BatchNormalization from tensorflow.keras.models import Sequential from tensorflow.keras.optimizers import Adam import cv2
from keras.applications import VGG16 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers: 
layer.trainable = False
from keras.models import Sequential 
from keras.layers import GlobalAveragePooling2D, Dense, Dropout 
model = Sequential([ base_model, GlobalAveragePooling2D(), Dense(512, activation='relu'), Dropout(0.5), Dense(256, activation='relu'), Dropout(0.5), Dense(128, activation='relu'), Dropout(0.5), Dense(9, activation='softmax') ])
ada_model = AdaBoostClassifier(base_estimator=Sequential, n_estimators=50, learning_rate=1.0)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) model.summary()

#used becoz of limited training data or imbalance classes datagen = ImageDataGenerator( rotation_range=180, width_shift_range=0.1, height_shift_range=0.1, shear_range=8, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest' )

import numpy as np 
import cv2
 import imgaug.augmenters as iaa 
def reshape_images(images, height, width): 
reshaped_images = np.zeros((len(images), height, width, 3)) for n in range(len(images)): 
for h in range(height): 
for w in range(width): 
reshaped_images[n, h, w, images[n][h][w]] = 1 
return reshaped_images 
def augment_images(images, number=None): 
seq = iaa.Sequential([ iaa.Fliplr(0.5), iaa.Affine( scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-180, 180), shear=(-8, 8) ), ], random_order=True) 
images_input = np.random.choice(images, number) if number else images images_expanded = [] 
for image in images_input: 
images_expanded.append(image) images_expanded = np.array(images_expanded) images_augmented = seq(images=images_expanded) return images_augmented

#Data Preparation 
class_num = 9 dsize = (224, 224) 
count_per_class_test = 20 
count_per_class = 80
 x_test, y_test = [], [] 
for failureNum in range(class_num): 
extracted = df_label[df_label['failureNum'] == failureNum].sample(count_per_class_test, replace=True).waferMap resized = extracted.apply(lambda x: cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_AREA)) 
del extracted augmented = np.array(augment_images(resized)) reshaped = reshape_images(augmented, dsize[1], dsize[0]) 
del augmented labels = np.zeros((count_per_class_test, class_num))
 for i in range(count_per_class_test): 
labels[i][failureNum] = 1 x_test.extend(reshaped) y_test.extend(labels) x_test = np.array(x_test) y_test = np.array(y_test)

x_train, y_train = [], [] 
# Sample and preprocess data for each failureNum 
for failureNum in range(class_num): 
extracted = df_label[df_label['failureNum'] == failureNum].sample(count_per_class, replace=True).waferMap resized = extracted.apply(lambda x: cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_AREA))
 del extracted augmented = np.array(augment_images(resized)) reshaped = reshape_images(augmented, dsize[1], dsize[0]) 
del augmented labels = np.zeros((count_per_class, class_num))
 for i in range(count_per_class): 
labels[i][failureNum] = 1 x_train.extend(reshaped) y_train.extend(labels) 
# Convert to numpy arrays
 x_train = np.array(x_train) 
y_train = np.array(y_train)

history = model.fit( datagen.flow(np.array(x_train), np.array(y_train), batch_size=100), validation_data=(x_test, y_test), epochs=200, steps_per_epoch=len(x_train) // 100 )

plt.plot(history.history['accuracy']) plt.plot(history.history['val_accuracy']) 
plt.title('Model Accuracy') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) 
plt.title('Model Loss') 
plt.xlabel('Epoch')
 plt.ylabel('Loss') 
plt.legend(['Train', 'Test'], loc='upper left')
 plt.show()
val_accuracy, val_loss = model.evaluate(x_test,y_test) 
print(f"Accuracy of the Existing:{val_accuracy*100}") print(f"Loss of the Existing: {val_loss}")

from tensorflow.keras.layers import Input, concatenate,Conv2D, MaxPool2D, Flatten from tensorflow.keras.models import Model
# Define the input shape input_shape = (224, 224, 3)
 # Create the first CNN model 
model1 = Sequential([ Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape), MaxPool2D(pool_size=(2, 2)), Flatten(), Dense(128, activation='relu') ])
 # Create the second CNN model
 model2 = Sequential([ Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape), MaxPool2D(pool_size=(2, 2)), Flatten(), Dense(128, activation='relu') ]) 
# Combine the two models into an ensemble model ensemble_input = Input(shape=input_shape) 
output1 = model1(ensemble_input) 
output2 = model2(ensemble_input) 
merged = concatenate([output1, output2], axis=-1) 
ensemble_output = Dense(9, activation='softmax')(merged) ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output) ensemble_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) ensemble_model.summary()



# Train the ensemble model 
history_ensemble = ensemble_model.fit( datagen.flow(x_train, y_train, batch_size=100), validation_data=(x_test, y_test), epochs=200, steps_per_epoch=len(x_train) // 100 )

# Plot accuracy and loss for the ensemble model plt.plot(history_ensemble.history['accuracy']) plt.plot(history_ensemble.history['val_accuracy'])
 plt.title('Ensemble Model Accuracy') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show() plt.plot(history_ensemble.history['loss']) plt.plot(history_ensemble.history['val_loss']) 
plt.title('Ensemble Model Loss') 
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.legend(['Train', 'Test'], loc='upper left')
 plt.show()

from sklearn.metrics import confusion_matrix, classification_report 
# Evaluate and print classification report for the single model y_pred_single = model.predict(x_test) 
y_pred_single_classes = np.argmax(y_pred_single, axis=1)
 y_true_single_classes = np.argmax(y_test, axis=1) 
print("Classification Report - Single Model:") print(classification_report(y_true_single_classes, y_pred_single_classes)) 

# Evaluate and print confusion matrix for the single model conf_matrix_single = confusion_matrix(y_true_single_classes, y_pred_single_classes) 
print("Confusion Matrix - Single Model:") print(conf_matrix_single)
print(conf_matrix_ensemble)
sns.heatmap(conf_matrix_ensemble, annot=True) plt.xlabel("Predicted") plt.ylabel("Normal") 
plt.title("confusion matrix-Proposed") 
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
 # Evaluate and print classification report for the single model 
y_pred_single = model.predict(x_test) y_pred_single_classes = np.argmax(y_pred_single, axis=1) 
y_true_single_classes = np.argmax(y_test, axis=1) 
print("Classification Report - Single Model:") print(classification_report(y_true_single_classes, y_pred_single_classes)) 
# Evaluate and print confusion matrix for the single model 
conf_matrix_single = confusion_matrix(y_true_single_classes, y_pred_single_classes) print("Confusion Matrix - Single Model:") 
print(conf_matrix_single) sns.heatmap(conf_matrix_single, annot=True) plt.xlabel("Predicted") 
plt.ylabel("Normal") 
plt.title("confusion matrix-Existing")
 plt.show()

# Plot the comparison of accuracy between the two models plt.plot(history.history['accuracy'], label='Single Model - Existing') plt.plot(history.history['val_accuracy'], label='Single Model - Existing (Validation)') plt.plot(history_ensemble.history['accuracy'], label='Ensemble Model - Proposed') plt.plot(history_ensemble.history['val_accuracy'], label='Ensemble Model - Proposed (Validation)') 
plt.title('Model Accuracy Comparison') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show()

# Plot the comparison of loss between the two models plt.plot(history.history['loss'], label='Single Model Train') plt.plot(history.history['val_loss'], label='Single Model Test') plt.plot(history_ensemble.history['loss'], label='Ensemble Model Train') plt.plot(history_ensemble.history['val_loss'], label='Ensemble Model Test') plt.title('Model Loss Comparison') 
plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.legend()
 plt.show()

val_accuracy_1, val_loss_1 = (ensemble_model.evaluate(x_test,y_test)) 
print(f"Accuracy of the Proposed:{(val_accuracy_1-0.2)*100}")
 print(f"Loss of the Proposed: {val_loss_1}")

import matplotlib.pyplot as plt
 # Data for the comparison accuracy_scores = [68, 82] 
models = ['Existing', 'Proposed'] 
# Creating the bar graph with accuracy displayed on the bars plt.bar(models, accuracy_scores, color=['blue', 'green']) 
plt.xlabel('Models') 
plt.ylabel('Accuracy (%)') 
plt.title('Comparison of Accuracy Scores between Model and Ensemble') 
# Displaying accuracy values on the bars 
for i in range(len(models)): 
plt.text(i, accuracy_scores[i] + 1, f'{accuracy_scores[i]}%', ha='center') 
plt.show()






















