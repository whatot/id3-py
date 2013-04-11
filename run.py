# -*- coding:utf-8 -*-

import sys
from dtree import *

"""
调用ID3算法的主函数是run_app。

运行
python -u run.py train.dat test.dat

数据格式

create_decision_tree(examples, attributes, target_attribute, heuristic_funtion) 
接受如下输入:

examples (训练or测试数据集) : list of dicts (python字典)
attributes : list
target_attribute: string
heuristic_funtion:  指向"gain"函数的函数指针

数据集文件最后一列为最终决定属性
"""


def get_training_file():
    """
    从命令行提取第一个参数,否则要求输入训练数据集文件名
    """

    if len(sys.argv) < 3:
        print "Please enter the training file: ",
        training_filename = sys.stdin.readline().strip()
    else:
        training_filename = sys.argv[1]

    try:
        fTrainIn = open(training_filename,"r")
    except IOError:
        print "Error: Could not find the training file specified or unable to open it" %training_filename
        sys.exit(0)

    return fTrainIn


def get_test_file():
    """
    从命令行提取第二个参数,否则要求输入训练数据集文件名
    """

    if len(sys.argv) < 3:
        print "Please enter the test file: ",
        test_filename = sys.stdin.readline().strip()
    else:
        test_filename = sys.argv[2]

    try:
        fTestIn = open(test_filename,"r")
    except IOError:
        print "Error: Could not find the test file specified or unable to open it" %test_filename
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

    for i in xrange(0, len(attrList)-1, 2):
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
    targetAttribute = "isresolved"
    #反转后，利用list.pop()去除最后一行，再反转回到原次序
    linesInTest.reverse()
    linesInTest.pop() #pop()弹出并返回最后一行
    linesInTest.reverse()

    attrList, attrDict = prepare_attributes(attributes)
    attrList.append(targetAttribute)
    #print attrList
    #['AreaName', 'VSAT', 'SupportVSAT', 'TimeToClose', 'ProductVSAT', 'isresolved']
    attrDict[targetAttribute] = 2 # since its a binary classification
    #print attrDict
    #{'ProductVSAT': '2', 'VSAT': '2', 'isresolved': 2, 'SupportVSAT': '2', 'AreaName': '2', 'TimeToClose': '2'}

    # prepare data
    testData = []
    for line in linesInTest:
        testData.append(dict(zip(attrList,[datum.strip() for datum in line.split("\t")])))

    #print testData
    #存放dicts的list
    #[......,{'ProductVSAT': '1', 'VSAT': '1', 'isresolved': '1', 'SupportVSAT': '0', 'AreaName': 'United_States', 'TimeToClose': '3'}]

    linesInTrain = [lineTrain.strip() for lineTrain in fTrainIn.readlines()]
    attributesTrain = linesInTrain[0].replace("\t"," ").split(" ")
    #print attributesTrain
    targetAttributeTrain = "isresolved"
    #once we have the attributes remove it from lines
    linesInTrain.reverse()
    linesInTrain.pop() # pops from end of list, hence the two reverses
    linesInTrain.reverse()

    attrListTrain, attrDictTrain = prepare_attributes(attributesTrain)
    attrListTrain.append(targetAttributeTrain)
    #print attrListTrain
    #['AreaName', 'VSAT', 'SupportVSAT', 'TimeToClose', 'ProductVSAT', 'isresolved']
    attrDictTrain[targetAttributeTrain] = 2 # since its a binary classification

    # prepare data
    trainData = []
    for lineTrain in linesInTrain:
        trainData.append(dict(zip(attrListTrain,[datum.strip() for datum in lineTrain.split("\t")])))

    #print trainData
    #[....{}, {}, {'ProductVSAT': '1', 'VSAT': '0', 'isresolved': '1', 'SupportVSAT': '1', 'AreaName': 'United_States', 'TimeToClose': '4'}]


    trainingTree = create_decision_tree(trainData, attrListTrain, targetAttributeTrain, gain)
    trainingClassification = classify(trainingTree, trainData)

    testTree = create_decision_tree(testData, attrList, targetAttribute, gain)
    testClassification = classify(testTree, testData)

    # also returning the example classification in both the files
    givenTestClassification = []
    for row in testData:
        givenTestClassification.append(row[targetAttribute])

    givenTrainClassification = []
    for row in trainData:
        givenTrainClassification.append(row[targetAttributeTrain])

    return trainingTree, trainingClassification, testClassification, givenTrainClassification, givenTestClassification


def accuracy(algoclassification, targetClassification):
    matching_count = 0.0

    for alg, target in zip(algoclassification,targetClassification):
        if alg == target:
            matching_count += 1.0

    # print len(algoclassification)
    # print len(targetClassification)
    return (matching_count / len(targetClassification)) * 100

def print_tree(tree, str):
    """
    打印树
    """

    if type(tree)== dict:
        #print "%s%s = " % (str,tree.keys()[0]),
        for item in tree.values()[0].keys():
            print "%s%s = %s" % (str, tree.keys()[0],item),
            print " : "
            print "|",
            print_tree(tree.values()[0][item],str + "  ")
    else:
        print "%s : %s" % (str, tree)

if __name__ == "__main__":

    fTrainIn = get_training_file()
    fTestIn = get_test_file()
    trainingTree, trainingClassification, testClassification, givenTrainClassification, givenTestClassification = run_app(fTrainIn,fTestIn)
    print_tree(trainingTree,"")
    print " Accuracy of training set (%s instances) : " % len(givenTrainClassification),
    print accuracy(trainingClassification, givenTrainClassification)
    print " Accuracy of test set (%s instances) : " % len(givenTestClassification),
    print accuracy(testClassification, givenTestClassification)




