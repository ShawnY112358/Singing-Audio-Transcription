import librosa
import torchcrepe
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

MAX_NOTE_NUM=sys.maxsize

# frame_length = 2048
# hop_length = frame_length // 4
frame_length = 1024
hop_length=512
fmin=librosa.note_to_hz('C2')
fmax=librosa.note_to_hz('C7')
silent_threshold = 1

def pitch_predict(model='yin'):
    onset_dir = './onset_predict'
    wav_dir = './data_test'
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for list in os.listdir(wav_dir):
        wav_path = os.path.join(wav_dir, list)
        with open(os.path.join(onset_dir, list.split('.')[0] + '.txt')) as f:
            onset_list = f.read().split('\n')[:-1]
            onset_list = [float(onset_list[i]) for i in range(len(onset_list))]
            f.close()
            pitch, offset = [], []
            if model == 'yin':
                y, sr, f0 = yin(wav_path)
            # f0 = [69 + 12 * math.log(f0[i] / 440, 2) for i in range(len(f0))]
            # onset, pitch = reprocess(onset_list, f0, sr)
            # offset = [0 for i in range(len(onset))]
            # with open(os.path.join(result_dir, list.split('.')[0] + '.txt'), 'w') as f:
            #     for i in range(len(onset)):
            #         f.write(str(onset[i]) + '\n' + str(offset[i]) + '\n' + str(pitch[i]) + '\n')
            #     f.close()
            onset, offset, onset_dict = note_cut(y, f0, sr, onset_list)
            pitch = mono_tri_quantile(onset_dict)
            pitch = [69 + 12 * math.log(pitch[i] / 440, 2) for i in range(len(pitch))]
            with open(os.path.join(result_dir, list.split('.')[0] + '.txt'), 'w') as f:
                for i in range(len(onset)):
                    f.write(str(onset[i]) + '\n' + str(offset[i]) + '\n' + str(pitch[i]) + '\n')
                f.close()

def yin(path):
    y, sr = librosa.load(path, sr=None)
    f0 = librosa.yin(y=y, sr=sr, frame_length=frame_length, hop_length=hop_length, fmin=fmin, fmax=fmax)
    return y, sr, f0

def note_cut(y, f0, sr, onset):
    time_unit = hop_length / sr
    onset_frame = [int(onset[i] / time_unit) for i in range(len(onset))]
    onset_frame.append(MAX_NOTE_NUM)

    #silent zone detect
    intervals = librosa.effects.split(y, top_db=silent_threshold, ref=np.max, frame_length=frame_length, hop_length=hop_length)
    silent_flag = [1 for i in range(len(f0))]
    for interval in intervals:
        silent_frame_start = int(interval[0] / (frame_length - hop_length))
        silent_frame_end = int(interval[1] / (frame_length - hop_length))
        for i in range(silent_frame_start, silent_frame_end + 1):
            silent_flag[i] = 0

    onset_dict = {}
    offset = []
    for i in range(len(onset)):
        onset_dict[onset[i]] = []
        for j in range(onset_frame[i], onset_frame[i + 1]):
            if j < len(f0) and silent_flag[j] == 1:
                onset_dict[onset[i]].append(float(f0[j]))
            else:
                offset.append((j + 0.5) * time_unit)
                break
            if j == onset_frame[i + 1] - 1:
                offset.append((j + 0.5) * time_unit)
        if onset_dict[onset[i]] == []:
            del onset_dict[onset[i]]
            offset.pop()
    onset = [key for key in onset_dict]
    return onset, offset, onset_dict

def mono_tri_quantile(onset_dict):
    pitch = []
    for key in onset_dict:
        avg = 0
        for i in range(int(len(onset_dict[key]) / 4), int(3 * len(onset_dict[key]) / 4) + 1):
            avg += onset_dict[key][i]
        avg /= int(3 * len(onset_dict[key]) / 4) - int(len(onset_dict[key]) / 4) + 1
        pitch.append(avg)

    return pitch

def reprocess(onset, f0, sr):
    pitch = []
    time_unit = hop_length / sr
    onset_frame = [int(onset[i] / time_unit) for i in range(len(onset))]    #每一个onset对应的f0下标
    onset_frame.append(len(f0) - 1)
    onset_dict = []
    for i in range(len(onset)):
        onset_dict.append([f0[j] for j in range(onset_frame[i], onset_frame[i + 1])])   #每一个onset到下一个onset之间的f0序列
    i = 1
    while True:
        if(i == len(onset_frame)):
            break;
        midi_list = onset_dict[i - 1]
        midi_l = len(midi_list)

        #step1
        static = 0
        static_len = 1
        static_zone = []
        zone_len = []
        for j in range(1, midi_l):
            if abs(midi_list[midi_l - j - 1] - midi_list[midi_l - j]) <= 1:
                static_len += 1
            else:
                if static_len >= 8:
                    zone_len.append(static_len)
                static_len = 1
            if static_len == 8:
                static += 1
                static_zone.append(midi_l - j - 1 + 7)  #start of the zone
        if static_len >= 8:
            zone_len.append(static_len)

        #step2
        if static == 0: #case 1
            if midi_l < 4:
                pitch.append(sum(midi_list) / midi_l)
            else:
                pitch.append(sum(midi_list[-5:]) / 4)
        elif static == 1:   #case 2
            pitch.append(static_est(static_zone[0], zone_len[0], midi_list))
        else:   #case 3 4 5
            static_midi = []
            for j in range(len(static_zone)):
                a = static_est(static_zone[j], zone_len[j], midi_list)
                static_midi.append(a)
            subs = [int(abs(static_midi[j] - static_midi[j - 1]) <= 1) for j in range(1, len(static_midi))]

            max_midi = max(static_midi)
            min_midi = min(static_midi)
            if sum(subs) == len(subs):   #case 3
                if len(static_zone) == 2:
                    pitch.append(max_midi)
                else:
                    sub = []
                    for j in range(1, len(static_midi)):
                        sub.append(static_midi[j] - static_midi[j - 1])
                    mono1 = [int(sub[j] >= 0) for j in range(len(sub))]
                    mono2 = [int(sub[j] <= 0) for j in range(len(sub))]
                    if(sum(mono1) == len(sub) or sum(mono2) == len(sub)):
                        pitch.append(max_midi)
                    else:
                        pitch.append(static_midi[0])
            else:   #case 4 5
                cut = 0
                for j in range(len(subs) - 1, -1, -1):
                    if subs[j] == 0 and zone_len[j + 1] >= 12 and zone_len[j] >= 12 and midi_l > static_zone[j] + 4: #case 5
                        onset_frame.insert(i, onset_frame[i - 1] + static_zone[j] + 1)
                        onset_dict.insert(i, midi_list[static_zone[j] + 1:])
                        pitch.append(static_est(static_zone[j], zone_len[j], midi_list))
                        onset.insert(i, (onset_frame[i] + 0.5) * time_unit)
                        cut = 1
                        break
                if cut == 1:
                    i += 1
                    continue
                else:   #case 4
                    pitch.append(static_midi[0])
        i += 1

    return onset, pitch

def static_est(zone_start, zone_len, midi):
    if zone_len <= 12:
        return sum(midi[zone_start - zone_len + 1:zone_start + 1]) / zone_len
    else:
        return sum(midi[zone_start - 12 + 1: zone_start + 1]) / 12



