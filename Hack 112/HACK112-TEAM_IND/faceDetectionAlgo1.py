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
  # plt.imshow(im)
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



tim=cv2.imread('LiveDemo2.jpg')
tim= cv2.cvtColor(tim, cv2.COLOR_BGR2GRAY)
tim= tim[10:700,400:900]
# plt.imshow(tim,plt.cm.gray)
tim=tim.flatten()
# plt.show()

tim= tim.reshape(-1,345000)

tim.shape



"""## Call the predict function"""

print(clsf.predict(tim))
Player = clsf.predict(tim)[0]


"""## Yipeeee!!! Its a correct prediction"""

