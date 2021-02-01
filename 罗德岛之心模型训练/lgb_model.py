import numpy as np
import pandas as pd
import lightgbm as lgb
import collections
import os
import sys


def predict(x_test, comments, model_input_path):
    '''
    预测得分并排序
    '''
    gbm = lgb.Booster(model_file=model_input_path)  # 加载model

    ypred = gbm.predict(x_test)

    predicted_sorted_indexes = np.argsort(ypred)[::-1]  # 返回从大到小的索引

    t_results = comments[predicted_sorted_indexes]  # 返回对应的comments,从大到小的排序

    return t_results

def validate(qids, targets, preds, k):
    """
    Predicts the scores for the test dataset and calculates the NDCG value.
    Parameters
    ----------
    data : Numpy array of documents
        Numpy array of documents with each document's format is [relevance score, query index, feature vector]
    k : int
        this is used to compute the NDCG@k

    Returns
    -------
    average_ndcg : float
        This is the average NDCG value of all the queries
    predicted_scores : Numpy array of scores
        This contains an array or the predicted scores for the documents.
    """
    query_groups = get_groups(qids)  # (qid,from,to),一个元组,表示这个qid的样本从哪到哪
    all_ndcg = []
    every_qid_ndcg = collections.OrderedDict()

    for qid, a, b in query_groups:
        predicted_sorted_indexes = np.argsort(preds[a:b])[::-1] # 从大到小的索引
        t_results = targets[a:b] # 目标数据的相关度
        t_results = t_results[predicted_sorted_indexes] #是predicted_sorted_indexes排好序的在test_data中的相关度
        dcg_val = dcg_k(t_results, k)
        idcg_val = ideal_dcg_k(t_results, k)
        ndcg_val = (dcg_val / idcg_val)
        all_ndcg.append(ndcg_val)
        every_qid_ndcg.setdefault(qid, ndcg_val)

    average_ndcg = np.nanmean(all_ndcg)
    return average_ndcg, every_qid_ndcg

def get_groups(qids):
    """Makes an iterator of query groups on the provided list of query ids.

    Parameters
    ----------
    qids : array_like of shape = [n_samples]
        List of query ids.

    Yields
    ------
    row : (qid, int, int)
        Tuple of query id, from, to.
        ``[i for i, q in enumerate(qids) if q == qid] == range(from, to)``

    """
    prev_qid = None
    prev_limit = 0
    total = 0

    for i, qid in enumerate(qids):
        total += 1
        if qid != prev_qid:
            if i != prev_limit:
                yield (prev_qid, prev_limit, i)
            prev_qid = qid
            prev_limit = i

    if prev_limit != total:
        yield (prev_qid, prev_limit, total)

def group_queries(training_data, qid_index):
    """
        Returns a dictionary that groups the documents by their query ids.
        Parameters
        ----------
        training_data : Numpy array of lists
            Contains a list of document information. Each document's format is [relevance score, query index, feature vector]
        qid_index : int
            This is the index where the qid is located in the training data

        Returns
        -------
        query_indexes : dictionary
            The keys were the different query ids and teh values were the indexes in the training data that are associated of those keys.
    """
    query_indexes = {}  # 每个qid对应的样本索引范围,比如qid=1020,那么此qid在training data中的训练样本从0到100的范围, { key=str,value=[] }
    index = 0
    for record in training_data:
        query_indexes.setdefault(record[qid_index], [])
        query_indexes[record[qid_index]].append(index)
        index += 1
    return query_indexes


def dcg_k(scores, k):
    """
        Returns the DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        DCG_val: int
            This is the value of the DCG on the given scores
    """
    return np.sum([
                      (np.power(2, scores[i]) - 1) / np.log2(i + 2)
                      for i in range(len(scores[:k]))
                      ])


def ideal_dcg_k(scores, k):
    """
    前k个理想状态下的dcg
        Returns the Ideal DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    # 相关度降序排序
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)

def plot_print_feature_importance(model_path):
    '''
    打印特征的重要度
    '''
    #模型中的特征是Column_数字,这里打印重要度时可以映射到真实的特征名

    if not os.path.exists(model_path):
        print("file no exists! {}".format(model_path))
        sys.exit(0)

    gbm = lgb.Booster(model_file=model_path)

    # 打印和保存特征重要度
    importances = gbm.feature_importance(importance_type='split')
    feature_names = gbm.feature_name()

    sum = 0.
    for value in importances:
        sum += value

    importance_data = []
    head = ['特征','重要度','重要度比例']

    for feature_name, importance in zip(feature_names, importances):
        # feat_id = int(feature_name.split('_')[1]) + 1
        print(' {} : {} : {}'.format(feature_name, importance, importance / sum))
        row = [feature_name,importance,importance/sum]
        importance_data.append(row)

    dff = pd.DataFrame(importance_data,columns=head)
    dff.to_csv('./重要度分析.csv',index=False)

def qid(id):
    num = id[0]
    counts = 0
    group_ids = []
    for new_num in id:
        if new_num == num:
            counts +=1
        else:
            num = new_num
            group_ids.append(counts)
            counts = 1
    group_ids.append(counts)
    return group_ids

# 读取数据
np.random.seed(16)
df = pd.read_excel('./data.xlsx')

qid_set = set(df['qid'])
qid_li = list(qid_set)
test_size = int(len(qid_li)*0.1)

# 筛除掉精二干员
level_li = [0,1,2]
df = df[df['level'].isin(level_li)]

test_li = np.random.choice(qid_li,size=test_size,replace=False)
test_set = set(test_li)

train_set = qid_set.difference(test_set)
train_li = list(train_set)

val_size = int(len(train_li)*0.2)
val_li = np.random.choice(train_li,size=val_size,replace=False)
val_set = set(val_li)

train_set = train_set.difference(val_set)
train_li = list(train_set)

train_data = df[df['qid'].isin(train_li)]
val_data = df[df['qid'].isin(val_li)]
test_data = df[df['qid'].isin(test_li)]


data = train_data.values
len_v = len(data[0])
train_y = data[:,-1]
train_X = data[:,1:len_v-1]
train_qid = data[:,0]
train_qid = train_qid.astype(int)
train_qid = qid(train_qid)
train_y = train_y.astype(int)

data2 = test_data.values
len_v = len(data2[0])
test_y = data2[:,-1]
test_X = data2[:,1:len_v-1]
test_qid = data2[:,0]
test_qid = test_qid.astype(int)
test_y = test_y.astype(int)

data3 = val_data.values
len_v = len(data3[0])
val_y = data3[:,-1]
val_X = data3[:,1:len_v-1]
val_qid = data3[:,0]
val_qid = val_qid.astype(int)
val_qid = qid(val_qid)
val_y = val_y.astype(int)

# 训练

# 训练集
train_data = lgb.Dataset(train_X, label=train_y, group=train_qid,feature_name=['star','level','class','subclass','power',
                                                                               '是否具有治疗', '是否具有支援', '是否具有输出',
                                                                               '是否具有减速', '是否具有生存', '是否具有防护', '是否具有削弱',
                                                                                '是否具有控场', '是否具有爆发',
            '挡二先锋练度比例','杀回先锋练度比例','投锋练度比例',
            '挡一近卫练度比例','挡二近卫练度比例','群体近卫练度比例',
            '远卫练度比例','快狙练度比例','群狙练度比例','超远狙练度比例',
            '特殊狙练度比例','盾练度比例','群奶练度比例','单奶练度比例',
            '减速练度比例','召唤练度比例','单法练度比例',
            '群法练度比例','快活练度比例','推拉练度比例','特殊干员练度比例'],
                         categorical_feature=['star','level','class','subclass','power'])

# 验证集
val_data = lgb.Dataset(val_X, label=val_y, group=val_qid,feature_name=['star','level','class','subclass','power',
                                                                       '是否具有治疗', '是否具有支援', '是否具有输出',
                                                                       '是否具有减速', '是否具有生存', '是否具有防护', '是否具有削弱',
                                                                        '是否具有控场', '是否具有爆发',
            '挡二先锋练度比例','杀回先锋练度比例','投锋练度比例',
            '挡一近卫练度比例','挡二近卫练度比例','群体近卫练度比例',
            '远卫练度比例','快狙练度比例','群狙练度比例','超远狙练度比例',
            '特殊狙练度比例','盾练度比例','群奶练度比例','单奶练度比例',
            '减速练度比例','召唤练度比例','单法练度比例',
            '群法练度比例','快活练度比例','推拉练度比例','特殊干员练度比例'],
                         categorical_feature=['star','level','class','subclass','power'])

# 参数设置
params = {
    'task': 'train',  # 执行的任务类型
    'boosting_type': 'gbrt',  # 基学习器
    'objective': 'lambdarank',  # 排序任务(目标函数)
    'metric': 'ndcg',  # 度量的指标(评估函数)
    # 'max_position': 16,  # @NDCG 位置优化
    'metric_freq': 1,  # 每隔多少次输出一次度量结果
    'train_metric': True,  # 训练时就输出度量结果
    'ndcg_at': [30],
    'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
    'num_iterations': 400,  # 迭代次数，即生成的树的棵数
    'learning_rate': 0.05,  # 学习率
    'num_leaves': 45,  # 叶子数
    # 'max_depth':14,
    'tree_learner': 'data',  # 用于并行学习，‘serial’： 单台机器的tree learner
    'nthread': 12,
    'min_data_in_leaf': 1,  # 一个叶子节点上包含的最少样本数量
    'feature_fraction': 0.7,
    'bagging_fraction': 0.65,
    'bagging_freq' : 4,
    # 'early_stopping_round':50,
    'verbose': 2  # 显示训练时的信息
}


gbm = lgb.train(params, train_data, valid_sets=[val_data],feature_name=['star','level','class','subclass','power',
                                                                        '是否具有治疗', '是否具有支援', '是否具有输出',
                                                                        '是否具有减速', '是否具有生存', '是否具有防护', '是否具有削弱',
                                                                        '是否具有控场', '是否具有爆发',
            '挡二先锋练度比例','杀回先锋练度比例','投锋练度比例',
            '挡一近卫练度比例','挡二近卫练度比例','群体近卫练度比例',
            '远卫练度比例','快狙练度比例','群狙练度比例','超远狙练度比例',
            '特殊狙练度比例','盾练度比例','群奶练度比例','单奶练度比例',
            '减速练度比例','召唤练度比例','单法练度比例',
            '群法练度比例','快活练度比例','推拉练度比例','特殊干员练度比例'],
                         categorical_feature=['star','level','class','subclass','power'])

# 保存模型
gbm.save_model('./model/model.txt')

# 预测
gbm = lgb.Booster(model_file='./model/model.txt')
test_predict = gbm.predict(test_X)
average_ndcg, _ = validate(test_qid, test_y, test_predict, 10)

# 所有qid的平均ndcg
print("all qid average ndcg: ", average_ndcg)
print("job done!")