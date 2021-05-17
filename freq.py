import torch
import torchcrepe
import os
import numpy as np
import librosa

def crepe_freq(path, model='tiny'):
    # Load audio
    audio, sr = torchcrepe.load.audio(path)

    # Here we'll use a 5 millisecond hop length
    hop_length = int(sr / 200.)

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 550

    # Select a model capacity--one of "tiny" or "full"
    #model = 'tiny'

    # Choose a device to use for inference
    device='cpu'
    
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 2048

    # Compute pitch using first gpu
    pitch = torchcrepe.predict(audio,
                               sr,
                               hop_length,
                               fmin,
                               fmax,
                               model,
                               batch_size=batch_size,
                               device=device)
    return pitch

# crepe
def test_crepe():
    dir = "../data_test"
    result_dir = '../crepe_freq'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for list in os.listdir(dir):
        path = os.path.join(dir, list)
        pitchs = crepe_freq(path).squeeze().numpy().tolist()
        with open(os.path.join(result_dir, list.split('.')[0] + '.txt'), 'w') as fp:
            for pitch in pitchs:
                fp.write(str(pitch))
                fp.write('\n')
            fp.close()
            print("%s done" % list)

#pYin
def test_pYin():
    dir = "../../data_test"
    result_dir = '../../pYin_freq'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for list in os.listdir(dir):
        y, sr = librosa.load(os.path.join(dir, list)) #音频信号值，采样率（默认22050）
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        with open(os.path.join(result_dir, list.split('.')[0] + '.txt'), 'w') as fp:
            f0 = f0.tolist()
            voiced_flag = voiced_flag.tolist()
            voiced_probs = voiced_probs.tolist()
            for i in range(len(f0)):
                fp.write(str(f0[i]) + ' ')
                fp.write(str(voiced_flag[i]) + ' ')
                fp.write(str(voiced_probs[i]) + '\n')
            print("%s done!" % list)
        fp.close()



