#id3-py

id3 decision tree using python

调用ID3算法的主函数是run.py中的 run_app。

###运行(暂时只支持python2.7)  
`python run.py train.dat test.dat`  
或者使用ori数据集  
`python run.py train-ori.dat test-ori.dat`


###数据格式  
`create_decision_tree(examples, attributes, target_attribute, heuristic_funtion)`

接受如下输入:

1. examples (训练or测试数据集) : list of dicts (python字典)
2. attributes : list
3. target_attribute: string
4. heuristic_funtion:  指向"gain"函数的函数指针  

##NOTE:数据集文件最后一列为最终决定属性

1. 数据集第一行为空格分行，并以2跟随在每个属性之后
`attr1 2 attr2 2 attr3 2 attr4 2 attr5 2 attr6 2 class 2`

2. 数据集数据部分全为tab(\t)分离
`1	1	0	0	0	0	0`
