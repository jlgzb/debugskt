import json
import numpy as np

def load_json_data(file_path):
    with open(file_path, 'r') as fin:
        list_result = json.load(fin)
    
    return list_result

def load_txt_data(file_path):
    with open(file_path, 'r') as fin:
        _list = [int(x.strip()) for x in fin.readlines()]
        
        return _list
    

# by gzb: ori code
def top_k_accuracy(scores, labels, topk=(2, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class. # by gzb: N 60
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k. # by gzb: [top1_score, topK_score]
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1] 
        # by gzb: shape of max_k_preds: N topk
        # by gzb: max_k_preds[:, 0] is index (actually is label) of top1 score; 
        # by gzb: max_k_preds [:, 1] is index of top2 score
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res

def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc



def debug():
    json_path = './cls30-pos21-clip1.json'
    list_result = load_json_data(json_path)

    #print ("type of data: {}".format(type(list_result))) # list

    list_result = np.array(list_result)
    #print ("shape of data: {}".format(list_result.shape)) # 8219, 30

def main():
    json_path = './test-clip20-j-hrnet-cliplen100-gpu2-time5-bs32-lr01.json'
    txt_path = './cs_label_pyskl.txt'
    list_scores = load_json_data(json_path)
    list_label = load_txt_data(txt_path)
    #labels = np.array(list_label)[:, np.newaxis]

    #print (labels.shape)
    #special_labels = [0, 17, 18] # [0, 17, 18], [10, 11, 28, 29], [15, 16]
    #special_labels = [10, 11, 28, 29] # [0, 17, 18], [10, 11, 28, 29], [15, 16]
    special_labels = [15, 16] # [0, 17, 18], [10, 11, 28, 29], [15, 16]

    max_k_preds = np.argsort(list_scores, axis=1)[:, -2:][:, ::-1]
    #print ("length of list_scores: {}".format(len(list_scores)))
    #print ("length of max_k_preds: {}".format(len(max_k_preds)))

    # get the corresponding prob
    #list_results = []
    for idx, key in enumerate(max_k_preds):
        #if list_label[idx] in key: # top 2, false pred
        #    continue

        #if list_label[idx] == key[0]: # top 1, false pred: skip true pred
        #    continue

        #if list_label[idx] != key[0]: # top1, true pred: skip false pred
        #    continue

        #if (list_label[idx] not in special_labels) or (key[0] not in special_labels):
        #    continue

        if (list_label[idx] != 11): # or (key[0] != ):
            continue
        if (key[0] != 11):
            continue

        top_1 = key[0]
        top_2 = key[1]
        #print ('key is: {}'.format(key))
        #print ('top_1 is: {}'.format(top_1))
        #print ('top_2 is: {}'.format(top_2))
        top_1_prob = list_scores[idx][top_1]
        top_2_prob = list_scores[idx][top_2]
        #print ('top1: {}; top2: {}; top1_prob: {}; top2_prob: {}.'. \
        #    format(top_1, top_2, top_1_prob, top_2_prob))

        # top k
        #if list_label[idx] != key[1]: # top 1, false pred: skip true pred
        #    continue


        print ('{}, {}, {}, {}, {}, {}'.format(list_label[idx], key[0], key[1], '#####', top_1_prob, top_2_prob))

    #print (idx)
    
    #print (max_k_preds[:10])


def main2():
    json_path = './test-clip20-j-hrnet-cliplen100-gpu2-time5-bs32-lr01.json'
    txt_path = './cs_label_pyskl.txt'
    list_scores = load_json_data(json_path)
    list_label = load_txt_data(txt_path)
    #labels = np.array(list_label)[:, np.newaxis]

    print (len(list_scores), len(list_label))

    #prob_ar = np.array(list_scores)
    #print (prob_ar.shape)

    #max_k_preds = np.argsort(list_scores, axis=1)[:, -2:][:, ::-1]
    #res = top_k_accuracy(list_scores, list_label)
    #print (res)

    res = top_k_accuracy(list_scores, list_label)
    print (res)

    total = len(list_label)
    _true = 0
    _false = 0

    max_k_preds = np.argsort(list_scores, axis=1)[:, -2:][:, ::-1]
    for idx, key in enumerate(max_k_preds):
        top_1 = key[0]
        top_2 = key[1]
        #print ('key is: {}'.format(key))
        #print ('top_1 is: {}'.format(top_1))
        #print ('top_2 is: {}'.format(top_2))
        top_1_prob = list_scores[idx][top_1]
        top_2_prob = list_scores[idx][top_2]
        #print ('top1: {}; top2: {}; top1_prob: {}; top2_prob: {}.'. \
        #    format(top_1, top_2, top_1_prob, top_2_prob))

        if top_1 in [10, 11, 28, 29]:
            #tmp = (top_1_prob + top_2_prob)/2
            if top_1_prob < (top_2_prob * 2 * 1.02 ):
                top_1 = key[1]
                top_2 = key[0]


        # top k
        #print ('{}, {}, {}, {}, {}, {}'.format(list_label[idx], key[0], key[1], '#####', top_1_prob, top_2_prob))

        if top_1 == list_label[idx]:
            _true += 1
        else:
            _false += 1

    acc = _true / total * 100

    print ('true: {}; false: {}'.format(_true, _false))
    print ('acc: {}%'.format(acc))




def statis_labels():
    txt_path = './cs_label_pyskl.txt'
    list_label = load_txt_data(txt_path)

    dict_label = {}
    for _label in list_label:
        if _label not in dict_label.keys():
            dict_label[_label] = 1
        else:
            dict_label[_label] += 1
    
    new_dict = sorted(dict_label.items())

    for v in new_dict:
        print (v)


if __name__ == '__main__':
    print ("Begin executing !")
    #print ('label, top1, top2, top1-prob, top2-prob')
    #main()
    main2()

    #statis_labels()

