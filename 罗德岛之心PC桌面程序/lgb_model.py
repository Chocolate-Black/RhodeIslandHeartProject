import pandas as pd
import lightgbm as lgb

def rank_result(data):
    return data['score']

class Model(object):
    def __init__(self):
        self.model = lgb.Booster(model_file='./model/model.txt')
        self.df =pd.read_excel('./ID/ID.xlsx')


    def predict(self,rec):
        data = {'success':False}
        if len(rec) == 0:
            return data

        n = len(rec)  # 总数
        # 计算子职业练度比例
        subclass_rate_group = [0 for _ in range(21)]

        for i in range(n):
            name = rec[i]['name']
            values = self.df[self.df['name'] == name].values
            subclass = values[0, 3]
            power = values[0, 4]
            level = rec[i]['level']
            subclass_rate_group[subclass] += (power / 3) * (level / 3)
        sum_value = sum(subclass_rate_group)
        if sum_value != 0:
            for i in range(len(subclass_rate_group)):
                subclass_rate_group[i] /= sum_value
        # 形成特征，作为模型的输入
        X = []
        for i in range(n):
            name = rec[i]['name']
            values = self.df[self.df['name'] == name].values
            # 判断是否是精二干员，如果是，则暂修改为level为2
            if rec[i]['level'] == 3:
                row = [values[0, 1], 2, values[0, 2], values[0, 3],
                       values[0, 4], values[0, 5], values[0, 6],
                       values[0, 7], values[0, 8], values[0, 9],
                       values[0, 10], values[0, 11], values[0, 12], values[0, 13]]
            else:
                row = [values[0, 1], rec[i]['level'], values[0, 2], values[0, 3],
                       values[0, 4], values[0, 5], values[0, 6],
                       values[0, 7], values[0, 8], values[0, 9],
                       values[0, 10], values[0, 11], values[0, 12], values[0, 13]]

            row.extend(subclass_rate_group)
            X.append(row)
        results = self.model.predict(X)
        # 预测并返回相关度
        data['predictions'] = list()
        for i in range(n):
            if rec[i]['level'] == 3:
                results[i] = -99
            row = {'name': rec[i]['name'], 'score': results[i]}
            data['predictions'].append(row)
        data['success'] = True
        # 排序
        data['predictions'].sort(key=rank_result,reverse=True)
        return data