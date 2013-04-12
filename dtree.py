# -*- coding:utf-8 -*-
"""
ID3算法,创建决策数
"""

import math

def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq     = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)

    return data_entropy


def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq       = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset     = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)


def create_decision_tree(data, attributes, target_attr, fitness_func):
    """
    Returns a new decision tree based on the examples given.
    """
    data    = data[:]
    vals    = [record[target_attr] for record in data]
    #vals是最终属性所有值的list
    default = majority_value(data, target_attr)

    # 如果数据集为空，或除去最终属性外的属性list为空，返回default
    if not data or (len(attributes) - 1) <= 0:
        return default
    #如果数据集中全部记录拥有相同的值,返回此值
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr,
                                fitness_func)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}

        #通过best属性的各个取值分别创建子树
        for val in get_values(data, best):
            #为best当前值创建子树
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)

            # 将子树添加到树的相应位置
            # Add the new subtree to the empty dictionary object in our new 
            # tree/node we just created.
            tree[best][val] = subtree

    return tree


########################## Helper Functions #####################
"""
The following are helper funtions used in creating the decision tree. These are
courtsey http://onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html
"""

"""
This module holds functions that are responsible for creating a new
decision tree and for using the tree for data classificiation.
"""

def majority_value(data, target_attr):
    """
    创建一个包含data中target_attr所有值的列表，返回出现频率最高的值
    """
    data = data[:]
    return most_frequent([record[target_attr] for record in data])

def most_frequent(lst):
    """
    返回在list中出现频率最高的项目
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)

    return most_freq

def unique(lst):
    """
    去除列表中的重复项目
    """
    lst = lst[:]
    unique_lst = []

    # 遍历lst,并将每个值只加入unique_lst一次
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)

    # 返回去除重复值的list
    return unique_lst

def get_values(data, attr):
    """
    剪去重复数值,返回列表
    """
    data = data[:]
    return unique([record[attr] for record in data])

def choose_attribute(data, attributes, target_attr, fitness):
    """
    遍历所有属性,并返回有最高信息增益(或最低熵)的属性
    """
    data = data[:]
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = fitness(data, attr, target_attr)
        if (gain >= best_gain and attr != target_attr):
            best_gain = gain
            best_attr = attr

    return best_attr

def get_examples(data, attr, value):
    """
    返回数据集中满足attr值的记录list
    """
    data = data[:]
    rtn_lst = []

    if not data:
        return rtn_lst
    else:
        for record in data:
            if record[attr] == value:
                rtn_lst.append(record)

        return rtn_lst

def get_classification(record, tree):
    """
    This function recursively traverses the decision tree and returns a
    classification for the given record.
    """
    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    if type(tree) == type("string"):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        t = tree[attr][record[attr]]
        return get_classification(record, t)

def classify(tree, data):
    """
    Returns a list of classifications for each of the records in the data
    list as determined by the given decision tree.
    """
    data = data[:]
    classification = []

    for record in data:
        classification.append(get_classification(record, tree))

    return classification

