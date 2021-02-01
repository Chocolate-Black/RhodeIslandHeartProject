## 明日方舟精二Box推荐器：罗德岛之心，项目心得与经验

优先精二哪些干员，一直是困扰明日方舟萌新的一大问题。通常来说，萌新会选择在网上发帖求助，或者是在B站等平台寻找节奏榜等，以得到自己想要的答案。然而，发帖求助往往需要等待一段时间才能得到回复；而节奏榜只展现干员强度，**并不能给出精二顺序的建议**。

于是，我希望用人工智能学习经验丰富的玩家的建议，做成一个网站工具，萌新只需将自己的阵容box上传，便可以得到建议精二的干员及其精二优先度顺序。

从2020年7月13日在NGA发帖到现在，经过近半年的时间（其实中间鸽了一段时间orz……），罗德岛之心明日方舟BOX推荐器终于大功告成！下面我将分享我的经历与经验心得，希望能对各位有所帮助。

### 一、问题分析：干员Box与推荐系统

在开始之前，首先要分析Box推荐这一问题的本质。

输入是玩家的干员Box，Box由多个干员构成，输出是干员的排序。实际上，这一问题可以通过LTR算法解决。

LTR，全称为Learning To Rank，顾名思义，这一种（类）学习如何排序的算法。LTR算法在我们日常生活中的应用非常广泛，其中最重要的一个应用就是推荐系统。淘宝的商品推荐，知乎、B站等平台的内容推荐等等，几乎都离不开LTR算法。

为了更好地理解LTR算法，我们以电商平台为例进行讲解。用户在搜索栏输入关键词后，会出现一系列相关商品。用户输入的关键词就是**Query**，与关键词相关的商品就是**Document**。我们希望让用户最感兴趣的商品排在前面，从而提升用户体验。LTR算法的作用就是在用户输入关键词（Query）后，对所有与关键词相关的商品（Document）进行排序，让最满足用户需求的商品尽可能地排在前面。

那么干员Box推荐与电商平台的推荐系统有什么相似之处呢？其实，**玩家上传的Box就可以看作是一个Query，Box里的干员就可以看作是Document**。我们的目标就是对Box里的所有干员进行评分，**评分越高的干员越值得优先精二**。

### 二、数据采集：小程序与云开发

在理清思路后，接下来要做的就是采集数据。首先，我们要确定采集什么样的数据——很明显，是box里每个干员的信息（干员自身属性及等级），不过还不够。我们知道监督学习需要有目标值，在box干员推荐这一问题中，目标值是干员的优先度——那么优先度该如何规定呢？

由于我准备用pair-wise的方式（后面会介绍）构建模型，因此一开始我把精二优先度划分为5个等级：最值得优先精二的干员，精二优先度较高的干员，可以延后考虑精二的干员，可选择性精二的干员以及无需精二的干员（含已精二干员）。后来在数据处理的过程中发现，由于**玩家更看重前10个以内的干员的排序结果**，因此可以延后考虑精二的干员和可选择性精二的干员这两个等级可以合并，以提高模型的精度。

此外，我把干员的等级划分成了4个等级区间，具体如下：

- 0：未精一
- 1：<=精一40级
- 2：精一40级~精一满级
- 3：已精二

在确定优先度的划分以及采集数据的格式后，我们罗德岛之心项目组的小伙伴们一起开发了一个用于采集数据的微信小程序，类似于抽卡模拟器（可以在微信搜索ArkHeart）。这里我们选择的是**云开发**模式，它非常适合个人以及小团队，其优点具体如下：

- 基本无需后端，开发方便；
- 云开发可以提供一个额度限定内免费的数据库，如果要采集的数据不多，平日流量不大，那么云开发绝对是不二之选。

在开发完用于采集数据的小程序后，我再次在NGA发帖，寻求经验丰富的玩家志愿者帮助采集数据。为了保证玩家给出建议的质量，我设置的要求是博士等级必须大于等于70级。最后，我们总共采集了1300多份box，数据记录在6w多条。

采集的数据格式如下：

```json
{
	"_id":"1b64dd7b5f6a202f004c10b1321f4de4",
	"time":1600790575457,
	"box":[{"id":277,"name":"阿消","chs":"unchosen","class":"特种","level":0,"star":4},			{"id":187,"name":"嘉维尔","chs":"unchosen","class":"医疗","level":1,"star":4},			{"id":198,"name":"讯使","chs":"unchosen","class":"先锋","level":2,"star":4}]
}
```
### 三、模型构建：特征工程与LGB

在采集完数据后，接下来要做的就是先确定使用什么样的模型。之前提到的LTR算法共有三大类：Pointwise类、Pairwise类以及Listwise类。Pointwise类算法把排序问题近似转换为回归问题，**只考虑单一文档，而不考虑文档之间的关系**；Pairwise类算法强调**文档顺序关系**，把排序问题近似为分类问题；而Listwise类算法则是直接优化排序列表，输入为单条样本为一个**文档排列**。

前两类算法用的比较广泛，这里我选用Pairwise类算法，构建GBRT+LR模型。这里就不展开讲原理了，有兴趣的可以自己在知乎或者csdn等平台搜一下，或是看文末的参考资料。

在训练模型之前，还有一项重要任务——**特征工程**。特征工程对于机器学习至关重要，很大程度上决定了模型的上限。

**首先，我们要对数据进行清洗。** 这里一开始我犯了个大错误，对于小程序采集到的数据过于自信，觉得没有必要从头到尾对数据进行清洗。但实际上，系统采集到的数据很脏！有的box因为系统bug只有一个干员被选中或者重复出现，有的box因为某些厨力玩家显得特别离谱（例如蜜蜡优先度高于塞雷娅……），还有的可能是因为侠客导致某些幻神干员明明没精二却没有被选择。这些数据无疑会影响到模型的训练（实际上我在对数据完全清洗后，ndcg指标上升了5个百分点），因此我们必须要进行清洗！我的做法是，把某些系统问题或者选的过于离谱的box直接删除，而剩余一些存在侠客因素或者问题不是很大的box，我个人凭借着玩家经验做了些许修改。

之后，我们要确定选取哪些特征，这些特征要尽可能地与因变量有关。我们采集的数据，仅仅有干员名称、干员星级、干员等级、干员职业以及人工标注的优先度。这些特征还远远不够。**并且，我们不能把干员名称作为特征——因为明日方舟的卡池是在不断更新的。**如果把干员名称作为特征，那么每次出新干员，数据就得重新采集，模型就得重新训练，泛化能力非常差。

那么该选取什么特征呢？比较好想到的是干员的公招标签。此外，我们可以想一下我们是如何给萌新推荐干员的。

**首先，我们重点关注干员的自身定位以及其强度。** 这里的干员定位我们可以用子职业来表示，比如说：技能回费先锋/杀回先锋/投锋，普通盾/奶盾等。干员强度我们可以用单值离散变量来表示，0代表很弱，1代表弱，2代表强，3代表很强。

**其次，我们会考虑整个Box里不同定位干员的练度比例**。比如：如果萌新有一个精二的能天使，那么蓝毒、白金等五星对空狙的精二优先度就会小一些。

因此，这里我增加了子职业以及各个子职业练度比例的特征。首先子职业的划分如下：

- 先锋0：挡二先锋0，杀回先锋1，投锋2
- 近卫1：挡一近卫3，挡二近卫4，群卫5，远卫6
- 狙击2：快狙7，群狙8，超远狙9，特殊狙10
- 重装3：盾11
- 医疗4：群奶12，单奶13
- 辅助5：减速14，召唤15
- 术士6：单法16，群法17
- 特种7：快活18，推拉19
- 单独的特殊类：20

有人可能会问：你这分的不对啊？近卫里面还有法伤近卫，盾里面还有奶盾，你这分的不够细啊。实际上，子职业并不是分的越细越好，如果分的过细，那么很容易出现类别之间的样本数量过于失衡的情况，影响模型的精度。比方说，奶盾与普通盾，其实可以靠“治疗”标签区分，因此无需过于细分子职业类别。

那么怎么确定分的好不好呢？一方面我们可以看评价模型的指标——NDCG的大小。NDCG介于0~1之间，越接近于1，说明模型分类越精确。

除了NDCG之外，还有NDCG@k，k是人为设定的整数值。这两者分别有什么含义呢？简单讲，NDCG和NDCG@k都是评判一个模型排序正确程度的指标，**区别在于，NDCG看的是所有item也就是doc的排序正确程度，而NDCG@k只关心排名前k个文档的排序正确程度。**这里我遇到了一个坑。一开始，我想当然地以为，由于玩家重点关注前10个干员的排序，因此优化指标就设为NDCG@10。然而，这样训练出来的模型，虽说在预测6星干员方面效果很不错，但对于某些好用低星干员，以及已有同类高练干员无需再练同类干员的情况下，效果不是很理想。原因在于：优先度为4，3的干员在前10位占据了绝大多数位置，然而在整个box中，优先度为2，1的干员才是多数。NDCG@k的k设的过于靠前，会让一些强力干员的优先度过高，以至于在某些特殊场合下（例如拥有精二蓝毒和一个精一40级白金）下，本应是某些较为适合box但本身强度不是特别高的干员的优先度较低，而让一些强度较高但已有同类替代的干员优先度较高。因此，NDCG@k中的k不可以过小。最后我选取了NDCG@30，效果不错。

另一方面，我们可以看特征重要度。在训练完树模型后，我们可以查看模型里各个特征的重要性——某个特征的重要性越高，就说明该特征在模型训练中发挥了越重要的作用，反之亦然。

经过反复尝试，我选取了上述的干员子职业分类方式，并去除了群攻、位移、费用回复、支援机械、快速复活、召唤等标签。

此外，干员子职业练度比例的计算公式如下：
$$
某子职业练度比例=\frac{\sum (干员强度/3)*(属于该子职业干员等级区间/3)}{所有子职业练度和}
$$

同一Box下的干员共享子职业练度比例的特征。

### 四、代码讲解

GBRT+LR的pairwise类算法，我们可以用LGB去做。这里必须吹一下LGB，又快又准！

下面开始详细讲解一下代码（部分代码参考了https://github.com/jiangnanboy/learning_to_rank）。
首先，我们需要加载已经处理过的表格数据：

```python
import pandas as pd
import numpy as np

np.random.seed(233) # 设置种子
df = pd.read_excel('./data.xlsx')
print(df)
```

接下来，我们需要把整个数据集划分为训练集、验证集和测试集。这里的比例约为7：2：1。

```python
def qid(id):
    """
    输入表格数据的qid（query id）列，转换为符合lgb模型的格式
    """
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


qid_set = set(df['qid']) # 获取qid（query id）集合
qid_li = list(qid_set)
test_size = int(len(qid_li)*0.1) # 获得测试集大小

# 筛除掉精二干员，精二干员不参与模型训练
level_li = [0,1,2]
df = df[df['level'].isin(level_li)]

# 数据集划分，X代表输入的特征，y代表目标值（相关度）
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
```

之后我们把训练集和验证集的数据加载为LGB的格式。

```python
import lightgbm as lgb

train_data = lgb.Dataset(train_X, label=train_y, group=train_qid,feature_name=['star','level','class','subclass','power',
            '是否具有治疗','是否具有支援','是否具有输出',
            '是否具有减速','是否具有生存','是否具有防护','是否具有削弱',
            '是否具有控场','是否具有爆发',
            '挡二先锋练度比例','杀回先锋练度比例','投锋练度比例',
            '挡一近卫练度比例','挡二近卫练度比例','群体近卫练度比例',
            '远卫练度比例','快狙练度比例','群狙练度比例','超远狙练度比例',
            '特殊狙练度比例','盾练度比例','群奶练度比例','单奶练度比例',
            '减速练度比例','召唤练度比例','单法练度比例',
            '群法练度比例','快活练度比例','推拉练度比例','特殊干员练度比例'],
                         categorical_feature=['star','level','class','subclass','power'])

val_data = lgb.Dataset(val_X, label=val_y, group=val_qid,feature_name=['star','level','class','subclass','power',
            '是否具有治疗','是否具有支援','是否具有输出',
            '是否具有减速','是否具有生存','是否具有防护','是否具有削弱',
            '是否具有控场','是否具有爆发',
            '挡二先锋练度比例','杀回先锋练度比例','投锋练度比例',
            '挡一近卫练度比例','挡二近卫练度比例','群体近卫练度比例',
            '远卫练度比例','快狙练度比例','群狙练度比例','超远狙练度比例',
            '特殊狙练度比例','盾练度比例','群奶练度比例','单奶练度比例',
            '减速练度比例','召唤练度比例','单法练度比例',
            '群法练度比例','快活练度比例','推拉练度比例','特殊干员练度比例'],
                         categorical_feature=['star','level','class','subclass','power'])
```

之后设置参数，进行训练，并保存模型。

```python
#参数已经调过参
params = {
    'task': 'train',  # 执行的任务类型
    'boosting_type': 'gbrt',  # 基学习器
    'objective': 'lambdarank',  # 排序任务(目标函数)
    'metric': 'ndcg',  # 度量的指标(评估函数)
    'metric_freq': 1,  # 每隔多少次输出一次度量结果
    'train_metric': True,  # 训练时就输出度量结果
    'ndcg_at': [30],# 设ndcg@30用于优化模型
    'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
    'num_iterations': 400,  # 迭代次数，即生成的树的棵数
    'learning_rate': 0.05,  # 学习率
    'num_leaves': 45,  # 叶子数
    'tree_learner': 'serial',  # 用于并行学习，‘serial’： 单台机器的tree learner
    'nthread': 12, # 线程数
    'min_data_in_leaf': 1,  # 一个叶子节点上包含的最少样本数量
    'feature_fraction': 0.7, # 每次迭代中随机选择70%的参数来建树，用于抑制过拟合
    'bagging_fraction': 0.65, # 每次迭代时用的数据比例，用于抑制过拟合
    'bagging_freq' : 4, # 意味着每 7 次迭代执行bagging
    'verbose': 2  # 显示训练时的信息
}
gbm = lgb.train(params, train_data, valid_sets=[val_data],feature_name=['star','level','class','subclass','power',
            '是否具有治疗','是否具有支援','是否具有输出',
            '是否具有减速','是否具有生存','是否具有防护','是否具有削弱',
            '是否具有控场','是否具有爆发',
            '挡二先锋练度比例','杀回先锋练度比例','投锋练度比例',
            '挡一近卫练度比例','挡二近卫练度比例','群体近卫练度比例',
            '远卫练度比例','快狙练度比例','群狙练度比例','超远狙练度比例',
            '特殊狙练度比例','盾练度比例','群奶练度比例','单奶练度比例',
            '减速练度比例','召唤练度比例','单法练度比例',
            '群法练度比例','快活练度比例','推拉练度比例','特殊干员练度比例'],
                         categorical_feature=['star','level','class','subclass','power'])

gbm.save_model('./model/model.txt')
```

最后，我们可以在测试集上进行检验，并显示特征重要度。

```python
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
    qids:数据里的QueryID
    targets:目标值，真实值
    preds:预测值
    k: ndcg@k 里的k
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
    dff.to_csv('./importance.csv',index=False)

# 预测
gbm = lgb.Booster(model_file='./model/model.txt')
test_predict = gbm.predict(test_X)
average_ndcg, _ = validate(test_qid, test_y, test_predict, 10)

# 所有qid的平均ndcg
print("all qid average ndcg@10: ", average_ndcg)

# 保存特征重要度
plot_print_feature_importance('./model/model.txt')
```

最终我们的模型在测试集上的ndcg@10约为0.9237，效果不错。

### 四、PC端的制作
模型终归是模型，要想让用户使用，就必须做出可视化界面。目前我们和kkdy合作一起开发网页版和小程序，这几天寒假闲着没事，我自学了pyqt做了个桌面程序，实现了box推荐的功能。源代码已经放在了我的Github里~

### 五、总结

费时半年的项目，终于做完了！个人感觉用AI做这个罗德岛之心Box推荐系统并不是非常困难。实际上，罗德岛之心的这个模式可以同样套用在其他手游上，不过本人精力有限，就只做方舟的了。

此外，这套干员推荐系统，稍作修改的话或许也可以用在肉鸽活动上，感觉可以一搞，不过现在思路还没完全确定，先等下一次肉鸽活动再说吧。

如果你喜欢我的项目的话，还请到我的Github里点一颗星，谢谢各位了~



#### 参考资料

1.机器学习排序算法：RankNet to LambdaRank to LambdaMART: https://www.cnblogs.com/genyuan/p/9788294.html

2.利用lightGBM做LTR排序： https://github.com/jiangnanboy/learning_to_rank

3.策略算法工程师之路-排序模型(LTR)及应用： https://zhuanlan.zhihu.com/p/113302654

#### 文件说明

- lgb_model.py: 模型训练代码
- lgb_model.ipynb: 项目经验分享与心得
- data_xlsx:训练数据集
- model：存放模型