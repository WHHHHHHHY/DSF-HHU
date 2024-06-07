```python
import pandas as pd
import numpy as np
import random

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import IPython
import matplotlib as mpl
from matplotlib import pyplot as plt

# 绘图参数
font2 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 20,
}
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
```


```python
def get_DecisionTree(train_x, train_y, test_x, test_y, feature_names, class_names, criterion='entropy', min_samples_leaf=5):
    """
    input:
        train_x: 训练集
        train_y: 训练集
        test_x : 测试集
        test_y : 测试集
        criterion: 决策树方法
        min_samples_leaf: 最小节点个数
    """
    # 定义决策树
    dc_tree = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf)
    # 训练决策树
    dc_tree.fit(train_x, train_y)
    # 得到决策树的预测结果
    pred_y = dc_tree.predict(test_x)
    # 计算准确度
    accuracy = accuracy_score(test_y, pred_y)
    # 绘制决策树
    fig = plt.figure(figsize=(7, 10))
    IPython.display.clear_output()
    print(f"准确度为: {accuracy}")
    tree.plot_tree(dc_tree, filled=True,
                   feature_names=feature_names,
                   class_names=class_names)
```

# 鸢尾花(iris)数据集
| | sepal length | sepal width | petal length | petal width | target | label |
| ---- | ------------ | ----------- | ------------ | ----------- | ------------- | ------------- |
| 0 | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa | 0 |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa | 0 |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa | 0 |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa | 0 |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa | 0 |


```python
# 鸢尾花(iris)数据集
# 从csv文件中读取数据
data = pd.read_csv('./iris_data.csv')
# print(data.head())
# print("----------------------------------------------------------------------------")
# print(data.describe())

# 划分训练集和测试集
x = data.drop(['target','label'],axis=1)
y = data.loc[:,'label']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=0)
```

# ID3算法
## 信息熵计算
$$
\begin{aligned}
&Ent(D) = \sum_{k = 1}^{n} p_k \cdot \log_2 p_k\\
&k为D集合中label的某种种类 \\
&p_k为D集合中label为k物品数量占D集合中物品总数量个数
\end{aligned}
$$
## 信息增益计算
$$
\begin{aligned}
&Gain(D, a) = Ent(D) - \sum_{v = 1}^{n} \frac{|D^v|}{|D|} Ent(D^v) \\
&a为D集合中物品的某种属性 \\
&v为D集合中属性为a的某种种类\\
&D^v为D集合中所有属性为a且种类为v的物品集合\\
\end{aligned}
$$

每次对除label外各个属性计算信息增益, 选信息增益最大的属性作为划分依据


```python
# ID3
get_DecisionTree(train_x, train_y, test_x, test_y, list(data.columns[0:4]), list(data['target'].unique()), criterion='entropy', min_samples_leaf=5)
```

    准确度为: 0.8666666666666667
    


    
![png](https://raw.githubusercontent.com/WHHHHHHHY/DSF-HHU/main/2. DecisionTree/content/output_5_1.png)
    


# CART算法
## 基尼指数
$$
\begin{aligned}
&Gini(D) = \sum_{k = 1}^{n} \sum_{k' \not = k} p_k \cdot p_{k'} = 1 - \sum_{k = 1}^{n} p_k^2 \\
&k为D集合中label的某种种类\\
\end{aligned}
$$
## 属性基尼指数
$$
\begin{aligned}
&Gini_index(D, a) = \sum_{v = 1}^{n} \frac{|D^v|}{|D|} Gini(D^v) \\
&a为D集合中物品的某种属性 \\
&v为D集合中属性为a的某种种类\\
&D^v为D集合中所有属性为a且种类为v的物品集合\\
\end{aligned}
$$

选属性基尼指数最小的属性


```python
# CART
get_DecisionTree(train_x, train_y, test_x, test_y, list(data.columns[0:4]), list(data['target'].unique()), criterion='gini', min_samples_leaf=5)
```

    准确度为: 0.8666666666666667
    


    
![png](https://raw.githubusercontent.com/WHHHHHHHY/DSF-HHU/main/2. DecisionTree/content/output_7_1.png)

    


# 红酒数据集
| | alcohol | malic_acid | ash | alcalinity_of_ash | magnesium | total_phenols | flavanoids | nonflavanoid_phenols | proanthocyanins | color_intensity | hue | od280/od315_of_diluted_wines | proline | target | label | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 14.23 | 1.71 | 2.43 | 15.6 | 127.0 | 2.80 | 3.06 | 0.28 | 2.29 | 5.64 | 1.04 | 3.92 | 1065.0 | class_0 | 0 |
| 1 | 13.20 | 1.78 | 2.14 | 11.2 | 100.0 | 2.65 | 2.76 | 0.26 | 1.28 | 4.38 | 1.05 | 3.40 | 1050.0 | class_0 | 0 |
| 2 | 13.16 | 2.36 | 2.67 | 18.6 | 101.0 | 2.80 | 3.24 | 0.30 | 2.81 | 5.68 | 1.03 | 3.17 | 1185.0 | class_0 | 0 |
| 3 | 14.37 | 1.95 | 2.50 | 16.8 | 113.0 | 3.85 | 3.49 | 0.24 | 2.18 | 7.80 | 0.86 | 3.45 | 1480.0 | class_0 | 0 |
| 4 | 13.24 | 2.59 | 2.87 | 21.0 | 118.0 | 2.80 | 2.69 | 0.39 | 1.82 | 4.32 | 1.04 | 2.93 |  735.0 | class_0 | 0 |



```python
from sklearn.datasets import load_wine

# 导入红酒数据集
data = {}
for idx in range(len(load_wine().feature_names)):
    data[load_wine().feature_names[idx]] = load_wine().data[:, idx]
data["target"] = load_wine().target_names[load_wine().target]
data["label"] = load_wine().target

data = pd.DataFrame(data)
# print(data.head())
# print("----------------------------------------------------------------------------")
# print(data.describe())


# 划分训练集和测试集
x = data.drop(['label', 'target'],axis=1)
y = data.loc[:,'target']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
```


```python
# ID3
get_DecisionTree(train_x, train_y, test_x, test_y, list(load_wine().feature_names), ['0', '1', '2'], criterion='entropy', min_samples_leaf=5)
```

    准确度为: 0.9444444444444444
    


    
![png](https://raw.githubusercontent.com/WHHHHHHHY/DSF-HHU/main/2. DecisionTree/content/output_10_1.png)

    



```python
# CART
get_DecisionTree(train_x, train_y, test_x, test_y, list(load_wine().feature_names), ['0', '1', '2'], criterion='gini', min_samples_leaf=5)
```

    准确度为: 0.8611111111111112
    


    
![png](https://raw.githubusercontent.com/WHHHHHHHY/DSF-HHU/main/2. DecisionTree/content/output_11_1.png)

    



```python

```


```python

```
