# acoustic_attack_on_keyboard
Acoustic side-channel attack on keyboards. I.e predicting 



## Feature extraction 
Resnet18 us used to extract features from melspectograms. The data is then saved under the names 'data.pt' and 'label.py'

## classification

Reads data saved under the names 'data.pt' and 'label.py' and then classifies using svm.
