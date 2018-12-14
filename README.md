# Face_Recognition_Keras (ML)

**Overall: It is a simple Face Recognition project which used Keras and VGG16 (implement in Face.py) to implement**

***Face DataSet Download:** https://drive.google.com/drive/folders/1xCWLdzEZenmooSlPxQDr9MuH_KR9m6ES?usp=sharing*

**Remark: There are 39 people and 65 images for each person (Training Set/Testing Set: 35/30)**
---

**Process: Training Accuracy accelerates fast during 1~15 Epochs**

![image](https://github.com/KBLin1996/Face_Recognition_Keras-ML-/blob/master/First_15Epochs.PNG)

---

**Result (Epoch 30), Test Loss: 0.8576, Test Accuracy: 81.1%**

![image](https://github.com/KBLin1996/Face_Recognition_Keras-ML-/blob/master/30Epochs.PNG)
### ***Conclusion: Looks like it will be better if we decrease the Epochs (Epoch 15 Loss = 0.8256 v.s Epoch 30 Loss = 0.8576)***

**Result (Epoch 20), Test Loss: 0.8636, Test Accuracy: 77.36%**

![image](https://github.com/KBLin1996/Face_Recognition_Keras-ML-/blob/master/20Epochs.PNG)
### ***Conclusion: Even worse if we lower the Epoch (Epoch 30 Loss = 0.8576 v.s Epoch 35 Loss = 0.8636)***

**Result (Epoch 35), Test Loss: 0.7736, Test Accuracy: 82.97%**

![image](https://github.com/KBLin1996/Face_Recognition_Keras-ML-/blob/master/35Epochs.PNG)
### ***Conclusion: It is better if we increase the Epoch (Epoch 30 Loss = 0.8576 v.s Epoch 35 Loss = 0.7736)***

### ***Final Conclusion: Our Face Recognition project optimizes at about 35 Epochs***
