from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def weighted_acc(preds, truths, verbose):
    # https://www.aclweb.org/anthology/P17-1142.pdf
    preds = preds.view(-1)
    truths = truths.view(-1)

    # print(preds)
    # print(truths)

    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1

    w_acc = (tp * n / p + tn) / (2 * n)

    if verbose:
        fp = n - tn
        fn = p - tp
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n)

    return w_acc

def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    acc5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f1 = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc2 = accuracy_score(binary_truth, binary_preds)

    return mae, acc2, acc5, acc7, f1, corr


def eval_mosei_emo(preds, truths, threshold, verbose=False):
    '''
    CMU-MOSEI Emotion is a multi-label classification task
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    total = preds.size(0)
    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds = torch.sigmoid(preds)

    # auc_score = roc_auc_score(truths.numpy(), preds.numpy())

    # aucs = []
    # for emo_ind in range(num_emo):
    #     aucs.append(roc_auc_score(truths.numpy()[:, emo_ind], preds.numpy()[:, emo_ind]))
    # aucs.append(np.average(aucs))

    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    # aucs.append(roc_auc_score(truths, preds, labels=list(range(num_emo)), average='weighted'))
    aucs.append(np.average(aucs))

    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    # add a new class that represents no emotion
    # truths = torch.cat((truths, torch.zeros((total, 1))), dim=1)
    # preds = torch.cat((preds, torch.zeros((total, 1))), dim=1)
    # for i in range(len(truths)):
    #     if torch.sum(turths[i]).item() == 0:
    #         turths[i][-1] = 1
    #     if torch.sum(preds[i]).item() == 0:
    #         preds[i][-1] = 1

    # f1s = f1_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    # f1s.append(f1_score(truths, preds, labels=list(range(num_emo)), average='weighted'))

    accs = []
    f1s = []
    for emo_ind in range(num_emo):
        preds_i = preds[:, emo_ind]
        truths_i = truths[:, emo_ind]
        accs.append(weighted_acc(preds_i, truths_i, verbose=verbose))
        f1s.append(f1_score(truths_i, preds_i, average='weighted'))

    accs.append(np.average(accs))
    f1s.append(np.average(f1s))

    acc_strict = 0
    acc_intersect = 0
    acc_subset = 0
    for i in range(total):
        if torch.all(preds[i] == truths[i]):
            acc_strict += 1
            acc_intersect += 1
            acc_subset += 1
        else:
            is_loose = False
            is_subset = False
            for j in range(num_emo):
                if preds[i, j] == 1 and truths[i, j] == 1:
                    is_subset = True
                    is_loose = True
                elif preds[i, j] == 1 and truths[i, j] == 0:
                    is_subset = False
                    break
            if is_subset:
                acc_subset += 1
            if is_loose:
                acc_intersect += 1

    acc_strict /= total # all correct（对于每个数据，完整预测正确所有 emotion 的正确率）
    acc_intersect /= total # at least one emotion is predicted（对于每个数据，至少预测正确一个 emotion 的正确率）
    acc_subset /= total # predicted is a subset of truth（对于每个数据，至少预测正确一个 emotion，并且预测存在的 emotion 必须是 truth 的子集 的正确率）

    return accs, f1s, aucs, [acc_strict, acc_subset, acc_intersect]


def eval_iemocap(preds, truths):
    # emos = ["Neutral", "Happy", "Sad", "Angry"]
    '''
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    total = preds.size(0)
    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds_inds = torch.argmax(preds, dim=-1)
    preds = torch.zeros_like(preds)

    for i in range(total):
        preds[i, preds_inds[i]] = 1

    accs = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds[:, i]
        truth_i = truths[:, i]
        acc = torch.sum(pred_i == truth_i).item() / total
        f1 = f1_score(truth_i, pred_i, average='weighted')
        accs.append(acc)
        f1s.append(f1)

    accs.append(np.average(accs))
    f1s.append(np.average(f1s))

    return accs, f1s
