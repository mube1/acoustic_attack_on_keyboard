## Acoustic_attack_on_keyboard
Acoustic side-channel attack on keyboards. I.e predicting 



### Feature extraction 
Resnet18 is used to extract features from mel-spectograms. The data is then saved under the names 'data.pt' and 'label.py'.

### Evaluation classification

Reads data saved under the names 'data.pt' and 'label.py' and then classifies using svm.


### Inference


Inference using the trained model included in this repo. Change the model if you want to use your own. This is not an optimized implementation. 

``` python 
python3 inference recodring_audio.wav

```



