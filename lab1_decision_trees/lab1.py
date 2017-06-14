import dtree as d
import monkdata as m
import numpy as np
import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

## Assigment 1 calculate the entropy of the training datasets
print "entrypy of three monk dataset"
print d.entropy(m.monk1)
print d.entropy(m.monk2)
print d.entropy(m.monk3)

## Assigment 2 calculate information gain by each attributes

print "monk1 data, information gain by each attribute"
for i in range(6):
    print d.averageGain(m.monk1,m.attributes[i])

print "monk2 data, information gain by each attribute"
for i in range(6):
    print d.averageGain(m.monk2,m.attributes[i])

print "monk3 data, information gain by each attribute"
for i in range(6):
    print d.averageGain(m.monk3,m.attributes[i])


## Assigment 3 Build the full decision trees for all three Monk datasets
print "performance"
t1=d.buildTree(m.monk1, m.attributes)
print d.check(t1, m.monk1),d.check(t1, m.monk1test)

t2=d.buildTree(m.monk2, m.attributes)
print d.check(t2, m.monk2),d.check(t2, m.monk2test)

t3=d.buildTree(m.monk3, m.attributes)
print d.check(t3, m.monk3), d.check(t3, m.monk3test)

## Assigment 4 Plot the classification error on the test sets as a function of partition fraction

accuracy=0
fraction=0
for f in np.arange(0.1,1,0.1)
    monk1train, monk1val = partition(m.monk1, f)
    t=d.buildTree(m.monk1train, m.attributes)
    t_pruned=d.allPruned(t)
    for tree in t_pruned:
        if d.check(tree,monk1val)>accuracy:
            accuracy=d.check(tree,monk1val)
            final_tree=tree
            fraction=f

print accuracy,fraction