from train import onset_p
from train import train_conv
from onset_predict import onset_predict
from pitch_predict import pitch_predict
import os
from eval import onset_eval, note_eval
from data_processor import vertify_data_iterator
from train import train_lstm
from pick_threshold import main
#
# K = 10
#
# train_conv(256, 20)
#
# onset_p('./model0.pt', 0)
#
# onset_predict('./model0.pt')
pitch_predict()
precision, recall, f1 = onset_eval()
precision, recall, f1 = note_eval()
# print(precision)

main()