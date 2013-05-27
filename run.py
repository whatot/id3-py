# -*- coding:utf-8 -*-

import sys
import dtree

"""
调用ID3算法的主函数是run_app。

运行(支持python2.7, 3.x)
python2 run.py train.dat test.dat
python3 run.py train.dat test.dat
或者使用ori数据集
python2 run.py train-ori.dat test-ori.dat
python3 run.py train-ori.dat test-ori.dat


数据格式

dtree.create_decision_tree(examples, attributes, target_attribute,
                           heuristic_funtion)
接受如下输入:

examples (训练or测试数据集) : list of dicts (python字典)
attributes : list
target_attribute: string
heuristic_funtion:  指向"dtree.gain"函数的函数指针

数据集文件最后一列为最终决定属性
"""


def get_training_file():
    """
    从命令行提取第一个参数,否则要求输入训练数据集文件名
    """

    if len(sys.argv) < 3:
        print("Please enter the training file: ",)
        training_filename = sys.stdin.readline().strip()
    else:
        training_filename = sys.argv[1]

    try:
        fTrainIn = open(training_filename, "r")
    except IOError:
        print("Error: Could not find the training file specified or unable "
              "to open it" % training_filename)
        sys.exit(0)

    return fTrainIn


def get_test_file():
    """
    从命令行提取第二个参数,否则要求输入训练数据集文件名
    """

    if len(sys.argv) < 3:
        print("Please enter the test file: ",)
        test_filename = sys.stdin.readline().strip()
    else:
        test_filename = sys.argv[2]

    try:
        fTestIn = open(test_filename, "r")
    except IOError:
        print("Error: Could not find the test file specified or unable to "
              "open it" % test_filename)
        sys.exit(0)

    return fTestIn


def prepare_attributes(attrList):
    """
    Returns a list of attributes with the sizes removed and also returns a
    dict item with the attributes as key and the number
    """

    attrList = attrList[:]
    attrs = []
    attrsDict = {}

    for i in range(0, len(attrList)-1, 2):
        attrs.append(attrList[i])
        # set the value of attribute name as key to its number
        attrsDict[attrList[i]] = attrList[i+1]

    return attrs, attrsDict


def run_app(fTrainIn, fTestIn):
    """
    Runs the algorithm on the data
    """

    linesInTest = [line.strip() for line in fTestIn.readlines()]
    attributes = linesInTest[0].split(" ")

    #反转后，利用list.pop()去除最后一行，再反转回到原次序
    linesInTest.reverse()
    linesInTest.pop()    # pop()弹出并返回最后一行
    linesInTest.reverse()

    attrList, attrDict = prepare_attributes(attributes)
    targetAttribute = attrList[-1]

    # prepare testdata
    testData = []
    for line in linesInTest:
        testData.append(dict(list(zip(attrList,
                                      [datum.strip()
                                       for datum in line.split("\t")]))))

    linesInTrain = [lineTrain.strip() for lineTrain in fTrainIn.readlines()]
    attributesTrain = linesInTrain[0].replace("\t", " ").split(" ")

    #once we have the attributes remove it from lines
    linesInTrain.reverse()
    linesInTrain.pop()   # pops from end of list, hence the two reverses
    linesInTrain.reverse()

    attrListTrain, attrDictTrain = prepare_attributes(attributesTrain)
    targetAttrTrain = attrListTrain[-1]

    # prepare data
    trainData = []
    for lineTrain in linesInTrain:
        trainData.append(dict(list(zip(attrListTrain,
                                       [datum.strip()
                                        for datum in lineTrain.split("\t")]))))

    trainingTree = dtree.create_decision_tree(trainData, attrListTrain,
                                              targetAttrTrain, dtree.gain)
    trainingClassify = dtree.classify(trainingTree, trainData)

    testTree = dtree.create_decision_tree(testData, attrList, targetAttribute,
                                          dtree.gain)
    testClassify = dtree.classify(testTree, testData)

    # also returning the example Classify in both the files
    givenTestClassify = []
    for row in testData:
        givenTestClassify.append(row[targetAttribute])

    givenTrainClassify = []
    for row in trainData:
        givenTrainClassify.append(row[targetAttrTrain])

    return (trainingTree, trainingClassify, testClassify, givenTrainClassify,
            givenTestClassify)


def accuracy(algoClassify, targetClassify):
    matching_count = 0.0

    for alg, target in zip(algoClassify, targetClassify):
        if alg == target:
            matching_count += 1.0

    # print len(algoClassify)
    # print len(targetClassify)
    return (matching_count / len(targetClassify)) * 100


def print_tree(tree, str):
    """
    打印树
    print the tree
    """

    if isinstance(tree, dict):
        # print("%s%s = " % (str, list(tree.keys())[0]))
        for item in list(list(tree.values())[0].keys()):
            print("  |  %s%s = %s" % (str, list(tree.keys())[0], item))
            print_tree(list(tree.values())[0][item], str + "   ")
            # the space in 'str + "   "' affect Backspace between Sub-layer
    else:
        print("   -->  %s : %s" % (str, tree))
        # --> stand for the targetAttribute

if __name__ == "__main__":

    fTrainIn = get_training_file()
    fTestIn = get_test_file()
    (trainingTree, trainingClassify, testClassify, givenTrainClassify,
     givenTestClassify) = run_app(fTrainIn, fTestIn)
    print_tree(trainingTree, "")
    print(" Accuracy of training set (%s instances) :  %s"
          % (len(givenTrainClassify),
             accuracy(trainingClassify, givenTrainClassify)))
    print(" Accuracy of test set (%s instances) :  %s"
          % (len(givenTestClassify),
             accuracy(testClassify, givenTestClassify)))
