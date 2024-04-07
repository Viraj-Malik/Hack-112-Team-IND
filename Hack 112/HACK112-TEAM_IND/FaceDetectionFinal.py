import Image_Capture
Player_Image="captured_image.jpg"

import glob

path= '/Users/virajmalik/Desktop/Hack 112/faces'

files=glob.glob(path+'/*/*')

import cv2
import numpy as np
from matplotlib import pyplot as plt

len(files)

files[0].split('/')[-2]


labels=[]
data=[]

for i in range (0,len(files),5):
  im=cv2.imread(files[i])
  im= im[10:700,400:900]
  plt.imshow(im)
  im= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  # plt.show()

labels=[]
data=[]

for i in range (len(files)):
  im=cv2.imread(files[i])
  im= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  im= im[10:700,400:900]

  name=files[i].split('/')[-2]
  labels.append(name)
  data.append(im.flatten())


data=np.array(data)
labels=np.array(labels)
data.shape,labels.shape

from sklearn.neighbors import KNeighborsClassifier

clsf= KNeighborsClassifier(n_neighbors=7)
clsf.fit(data,labels)



tim=cv2.imread(Player_Image)
tim= cv2.cvtColor(tim, cv2.COLOR_BGR2GRAY)
tim= tim[10:700,400:900]
plt.imshow(tim,plt.cm.gray)
tim=tim.flatten()
plt.show()

tim= tim.reshape(-1,345000)

tim.shape



"""## Call the predict function"""

Pred1 = clsf.predict(tim)[0]


import glob
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load images
path = '/Users/virajmalik/Desktop/Hack 112/faces'
files = glob.glob(path + '/*/*')

# Read images and preprocess
data = []
labels = []

for file in files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = img[10:700, 400:900]  # Cropping
    data.append(img.flatten())
    labels.append(file.split('/')[-2])

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize SVM classifier
clf = SVC(kernel='linear')

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Predict label for a test image
test_image = cv2.imread(Player_Image, cv2.IMREAD_GRAYSCALE)
test_image = test_image[10:700, 400:900]  # Cropping
test_image = test_image.flatten()
test_image = test_image.reshape(1, -1)  # Reshape for prediction
predicted_label = clf.predict(test_image)


Pred2=predicted_label[0]

if Pred1==Pred2:
   Final_Player=str(Pred1)
   print(str(Final_Player))

else:
   Final_Player=None
   print("Sorry! Error Training Model")