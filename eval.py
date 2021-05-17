import os

K = 5

def onset_eval():
    tp_list, fp_list, fn_list = [], [], []
    predict_dir = './onset_predict'
    ground_truth_dir = '../ground_truth'
    test_num = 0
    for list in os.listdir(predict_dir):
        with open(os.path.join(predict_dir, list)) as f:
            predict = f.read().split('\n')
            predict = [float(predict[i]) for i in range(len(predict) - 1)]
            f.close()
            with open(os.path.join(ground_truth_dir, list.split('.')[0] + '.GroundTruth.txt')) as f:
                ground_truth = f.read().split('\n')
                ground_truth = [float(ground_truth[i * 3]) for i in range(int(len(ground_truth) / 3))]
                f.close()

                tp, fp, fn = 0, 0, 0
                i, j = 0, 0
                while i < len(predict) and j < len(ground_truth):
                    if(abs(predict[i] - ground_truth[j]) <= 0.05):
                        tp += 1
                        j += 1
                        i += 1
                    elif predict[i] > ground_truth[j]:
                        fn += 1
                        j += 1
                    else:
                        i += 1
                        fp += 1
                fp += len(predict) - i
                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)
        test_num += 1
        if test_num == int(len([l for l in os.listdir(predict_dir)]) / K):
            break

    precision = [tp_list[i] / (tp_list[i] + fp_list[i]) for i in range(len(tp_list))]
    recall = [tp_list[i] / (tp_list[i] + fn_list[i]) for i in range(len(tp_list))]
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]
    return precision, recall, f1

def note_eval():
    #non_detected, spurious, split, merge, correct, badly_detected
    tp_list, fp_list, fn_list = [], [], []

    predict_dir = './result'
    ground_truth_dir = '../ground_truth'
    test_num = 0
    for list in os.listdir(predict_dir):
        with open(os.path.join(predict_dir, list)) as f:
            predict = f.read().split('\n')
            predict = [float(predict[i]) for i in range(len(predict) - 1)]
            f.close()
            with open(os.path.join(ground_truth_dir, list.split('.')[0] + '.GroundTruth.txt')) as f:
                ground_truth = f.read().split('\n')
                ground_truth = [float(ground_truth[i]) for i in range(len(ground_truth))]
                f.close()

                tp, fp, fn = 0, 0, 0
                i, j = 0, 0
                while i * 3 < len(predict) and j * 3 < len(ground_truth):
                    if(abs(predict[i * 3] - ground_truth[j * 3]) <= 0.05
                            and abs(predict[i * 3 + 2] - ground_truth[j * 3 + 2]) <= 2):
                        tp += 1
                        j += 1
                        i += 1
                    elif predict[i * 3] > ground_truth[j * 3]:
                        fn += 1
                        j += 1
                    else:
                        i += 1
                        fp += 1
                fp += len(predict) / 3 - i
                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)
        test_num += 1
        if test_num == int(len([l for l in os.listdir(predict_dir)]) / K):
            break
    precision = [tp_list[i] / (tp_list[i] + fp_list[i]) for i in range(len(tp_list))]
    recall = [tp_list[i] / (tp_list[i] + fn_list[i]) for i in range(len(tp_list))]
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]
    return precision, recall, f1