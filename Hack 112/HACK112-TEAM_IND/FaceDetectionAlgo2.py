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
print("Accuracy:", accuracy)

# Predict label for a test image
test_image = cv2.imread('LiveDemo2.jpg', cv2.IMREAD_GRAYSCALE)
test_image = test_image[10:700, 400:900]  # Cropping
test_image = test_image.flatten()
test_image = test_image.reshape(1, -1)  # Reshape for prediction
predicted_label = clf.predict(test_image)
print("Predicted label:", predicted_label[0])
