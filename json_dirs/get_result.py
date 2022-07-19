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

def debug():
    json_path = './cls30-pos21-clip1.json'
    list_result = load_json_data(json_path)

    #print ("type of data: {}".format(type(list_result))) # list

    list_result = np.array(list_result)
    #print ("shape of data: {}".format(list_result.shape)) # 8219, 30

def main():
    json_path = './test-clip20-j-hrnet-cliplen100-gpu2-time5-bs32-lr01.json'
    txt_path = './NTU_CS_test_label_30.txt'
    list_scores = load_json_data(json_path)
    list_label = load_txt_data(txt_path)
    #labels = np.array(list_label)[:, np.newaxis]

    #print (labels.shape)
    special_labels = [10, 11, 28, 29] # [0, 17, 18], [10, 11, 28, 29], [15, 16]

    max_k_preds = np.argsort(list_scores, axis=1)[:, -2:][:, ::-1]
    #print ("length of list_scores: {}".format(len(list_scores)))
    #print ("length of max_k_preds: {}".format(len(max_k_preds)))

    # get the corresponding prob
    #list_results = []
    for idx, key in enumerate(max_k_preds):
        #if list_label[idx] in key: # top 2, false pred
        #    continue

        #if list_label[idx] == key[0]: # top 1, false pred
        #    continue

        if list_label[idx] != key[0]: # top1, true pred
            continue

        if (list_label[idx] not in special_labels) or (key[0] not in special_labels):
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
        print ('{}, {}, {}, {}, {}, {}'.format(list_label[idx], key[0], key[1], '#####', top_1_prob, top_2_prob))

    #print (idx)
    
    #print (max_k_preds[:10])




if __name__ == '__main__':
    #print ("Begin executing !")
    #print ('label, top1, top2, top1-prob, top2-prob')
    main()
