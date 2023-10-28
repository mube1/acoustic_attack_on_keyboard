###
This is a script to extract features that will serve for the prediction of characters from their keystroke sounds. The experiment is done on Dell KB216.

It assumes that there is a folder that contains sounds of keystrokes collected from the keyboard.
The output of this script is data.pt and label.npy which are tensors and labels respectively corresponding to each character sound.
Later on, the features in data.pt will be used by machine learning algorithms for the classification of the sounds.
###

  
import numpy as np
import os,librosa,
import torch
import torch.nn as nn

import torchvision.models as models 

"""

"""

## path to the folder where the audio files for the letters reside
path='dataset/'

# given a tensor of size between 0 and 224, this function will zero padd it into   (224x224)
auto_pad_tensor = lambda tensor:torch.nn.functional.pad(tensor, (224 - tensor.size()[-2:][0], 224 - tensor.size()[-2:][1]), "constant", 0)

# loop through the main folder referred with variable 'path', find the sub-folders named with characters
x,y=[],[]
for character_folder in os.listdir(path): # looop though the main folder
  files=os.listdir(path+character_folder) # looop though each characters folders to find all the audio files 
  for file in files:
    audio, sr = librosa.load(p+'/'+file) # load each file
    S=librosa.feature.melspectrogram(y=audio, sr=sr) # get the spectogram of each loaded each file
    S_dB = librosa.power_to_db(S, ref=np.max)
    padded=auto_pad_tensor(torch.tensor(S_dB)) # padd tensors into 224x224 to be fed to resnet network
    padded=padded.unsqueeze(0).unsqueeze(0).flatten() # reshape the padded output into 224x224 and not 224,224,1,1
    x.append(padded) # append each data on x
    y.append(character_folder) # append each label on y

x=torch.vstack(x) # create a larger torch file out of the collected list of tensors x
feartures=feature(x) # exract features with resnet, this might be difficult to run at once

torch.save(feartures,'data'.pt') # save data
np.save('label.npy',np.array(y)) # save label






