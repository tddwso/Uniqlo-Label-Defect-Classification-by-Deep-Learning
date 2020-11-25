

# In[1]:

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.utils import to_categorical


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train", required=True,
	help="path to input dataset")
ap.add_argument("-t", "--test", required=True,
	help="path to input dataset")

ap.add_argument("-m", "--model", type=str, default="MAS.model",
	help="path to output loss/accuracy plot")

import sys


sys.argv[1:] = '-d train -t test -m model'.split()
 
 
args = vars(ap.parse_args())


# ## Where AM I 資料讀入及前處理

# In[3]:


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["train"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label) 



from joblib import dump, load

dump(data, 'data.joblib')


#data = load('data.joblib')

dump(labels, 'labels.joblib')


#labels = load('labels.joblib')



# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0

labels = np.array(labels)

labels1=labels

#=========================================
""" 2 values for label
# perform one-hot encoding on the labels
lb = LabelBinarizer()

labels = lb.fit_transform(labels)
labels1 = labels

labels = to_categorical(labels)

"""

# Encoding categorical data more than 2 values
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

labels = labelencoder_X.fit_transform(labels)


labels = to_categorical(labels)

#X = onehotencoder.fit_transform(X).toarray()



#=======================================================
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")



# load the VGG16 network, ensuring the head FC layer sets are left
# off

baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

INIT_LR = 1e-3
EPOCHS = 20
BS = 8

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])



# train the head of the network
print("[INFO] training head...")

H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)



model.save("mas5.h5")

#========================================================
#呼叫已經訓練好的模型，不用作fit

from tensorflow.keras.models import load_model

model = load_model("mas5.h5")


H=np.load('my_history.npy',allow_pickle='TRUE').item()

#=============================================
#把訓練好的模型存起來or拿來使用

np.save('my_history.npy',H.history)

#np.save('my_history.npy',H)


H=np.load('my_history.npy',allow_pickle='TRUE').item()


#from tensorflow.keras.models import load_model

#model = load_model("mas5.h5")

##=============================================

# New data prediction
#new_prediction = classifier.predict(np.array([[1, -1, 0, 0, 1, 1, 1, -1]]))

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs2 = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs2
	))

# show a nicely formatted classification report
#print(classification_report(testY.argmax(axis=1), predIdxs2,	target_names=lb.classes_))



# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
#cm = confusion_matrix(testY.argmax(axis=1), predIdxs2)

#total = sum(sum(cm))
#acc = (cm[0, 0] + cm[1, 1] +cm[2, 2]) / total
#sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
#print(cm)
#print("acc: {:.4f}".format(acc))
#print("sensitivity: {:.4f}".format(sensitivity))
#print("specificity: {:.4f}".format(specificity))

#======================================================

arr = np.array(predIdxs2)
testIds2 = np.argmax(testY, axis=1)

#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(testIds2, arr)
print('Confusion Matrix\n')
print(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(testIds2, arr)))

print('Micro Precision: {:.2f}'.format(precision_score(testIds2, arr, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(testIds2, arr, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(testIds2, arr, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(testIds2, arr, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(testIds2, arr, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(testIds2, arr, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(testIds2, arr, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(testIds2, arr, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(testIds2, arr, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(testIds2, arr, target_names=['C2','C3','OK']))

#======================================================

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()

#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H["loss"], label="train_loss")

#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

#plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H["accuracy"], label="train_acc")

#plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on Unqulo Label")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")


#============================Label Check=============



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications import imagenet_utils

import imutils

imagePaths = list(paths.list_images(".\\test"))


for ii, imagePath in enumerate(imagePaths):	

  
    print("processing image {}/{}".format(ii + 1,
		len(imagePaths)))
          
      
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, (224, 224))
    image1=image
    
    image = img_to_array(image)
    
    image = np.expand_dims(image, axis=0)
    

    
    image = np.array(image) / 255.0
    
    #image = imagenet_utils.preprocess_input(image)
    
    result = model.predict(image)
    
    i = np.argmax(result,axis=1)
        
    #print(result)
    if i==0 :
        prediction = 'C2 defect'

               
    elif i==1:
        prediction = 'C3 defect'


    else:
        prediction = 'OK'
            
        
    print(" The test image is: ", prediction)     
    
    
    label = imagePath.split(os.path.sep)[-2]
    
    print(" The Actual image is :", label)
    
    image1 = imutils.resize(image1, height=700)
    
    cv2.imshow("Image is:", image1)
    
    cv2.waitKey(2000)
    cv2.destroyAllWindows()     

    
#====================================

import seaborn as sns

matrix = confusion_matrix(testY.argmax(axis=1), predIdxs2)

plt.figure(figsize=(10, 10))
sns.heatmap(matrix, annot=True, cbar=False, cmap="Blues",
            xticklabels=confusion,
            yticklabels=confusion)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
    
#================
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from scipy import interp
from sklearn.metrics import roc_auc_score

name = {
0:'C2',
1:'C3',
2:'OK'}

n_classes =labels.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], predIdxs[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), predIdxs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#Plot of a ROC curve for a specific class


for i in range(n_classes):
    print("ROC=",name[i])


for i in range(n_classes):
    plt.figure()
    lw = 2
    plt.plot(fpr[i], tpr[i], color='darkorange',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic {}'.format( name[i] ))
    plt.legend(loc="lower right")
    plt.show()

#==============================

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
# Finally average it and compute AUC
mean_tpr /= n_classes
    
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    s=name[i]
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(s, roc_auc[i]))
    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to Retina damages')
plt.legend(loc="lower right")
plt.show()
        
