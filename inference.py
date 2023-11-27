
# Inference example

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os, sys
import librosa
import torch
import torchvision.models as models 
import pickle
import numpy as np

# loading fature extactor
model = models.resnet18(pretrained=True)
model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
feature=nn.Sequential(*list(model.children())[:-1])

def auto_pad_tensor(tensor):
  width, height = tensor.size()[-2:]
  padding = (224 - width, 224 - height)
  return torch.nn.functional.pad(tensor, padding, "constant", 0)

path=sys.argv
audio_file = AudioSegment.from_wav(path)
audio_segments = split_on_silence(audio_file,min_silence_len=5)
files=[]

with open('svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)

def pre_process(audio):
   
   pass
for i, audio_segment in enumerate(audio_segments):
    name='./'+str(i)+'.wav'
    audio_segment.export(name, format="wav")
    y, sr = librosa.load(name)
    S=librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)        
    d=auto_pad_tensor(torch.tensor(S_dB) )
    files.append(d.unsqueeze(0).unsqueeze(0))
    os.remove(name)

x=torch.vstack(files)
out=feature(x)
out=out.squeeze(-1).squeeze(-1).detach().numpy()   

preds = svm_model.predict(out)
output=list( map(lambda x: '*' if x==1 else ' ' ))
print(''.join(output))
