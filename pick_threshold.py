from onset_predict import onset_predict
from eval import onset_eval
import matplotlib.pyplot as plt
from pitch_predict import pitch_predict
from eval import note_eval

def main():
    thresholds = [0.0001 + 0.0001 * i for i in range(100)]
    precisions, recalls, f1s = [], [], []
    # for threshold in thresholds:
    #     print(threshold)
    #     onset_predict('./model0.pt', threshold)
    #     pitch_predict()
    #     precision, recall, f1 = note_eval()
    #     precisions.append(sum(precision) / len(precision))
    #     recalls.append(sum(recall) / len(recall))
    #     f1s.append(sum(f1) / len(f1))
    #
    with open('precision_2.txt', 'r') as f:
        precisions_2 = f.read().split('\n')[:-1]
        precisions_2 = [float(precisions_2[i]) for i in range(len(precisions_2))]
        f.close()
    with open('recall_2.txt', 'r') as f:
        recalls_2 = f.read().split('\n')[:-1]
        recalls_2 = [float(recalls_2[i]) for i in range(len(recalls_2))]
        f.close()
    with open('f1_2.txt', 'r') as f:
        f1s_2 = f.read().split('\n')[:-1]
        f1s_2 = [float(f1s_2[i]) for i in range(len(f1s_2))]
        f.close()

    # with open('precision.txt', 'r') as f:
    #     precisions = f.read().split('.')[1:]
    #     precisions = [float('0.' + precisions[i]) for i in range(len(precisions))]
    #     f.close()
    # with open('recall.txt', 'r') as f:
    #     recalls= f.read().split('.')[1:]
    #     recalls = [float('0.' + recalls[i]) for i in range(len(recalls))]
    #     f.close()
    # with open('f1.txt', 'r') as f:
    #     f1s = f.read().split('.')[1:]
    #     f1s = [float('0.' + f1s[i]) for i in range(len(f1s))]
    #     f.close()

    plt.figure(1)
    plt.plot(thresholds, precisions_2, '-bo', label="Precision")
    plt.plot(thresholds, recalls_2, '-gx', label='Recall')
    plt.plot(thresholds, f1s_2, 'r-', label='F1-measure')
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()