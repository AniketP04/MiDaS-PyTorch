#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install timm')


# In[2]:


import cv2
import torch
import matplotlib.pyplot as plt
import timm


# In[3]:


midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()


# In[4]:


transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform


# In[5]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

        print(output)
    plt.imshow(output)
    cv2.imshow('CV2Frame', frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

plt.show()

