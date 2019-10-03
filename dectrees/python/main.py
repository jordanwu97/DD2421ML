import monkdata as m
from monkdata import Sample
import dtree
import drawtree_qt5

datasets = {
    "MONK-1": m.monk1,
    "MONK-2": m.monk2,
    "MONK-3": m.monk3,
}

testsets = {
    "MONK-1": m.monk1test,
    "MONK-2": m.monk2test,
    "MONK-3": m.monk3test,
}


# Part 3
for name, dset in datasets.items():

    print (f"{name} entropy: {round(dtree.entropy(dset),5)}")

## Part 4
def getMaxGain(dataset, attributes):

    gains = [ (a, dtree.averageGain(dataset, a)) for a in attributes ]
    
    g = [ str(round(b, 5)) for a,b in gains ]

    print (" & ".join(g))

    attribute_max = max(gains, key=lambda x: x[1])

    print ("Max gain on attribute", attribute_max)

    return attribute_max

for name, dset in datasets.items():
    print (f"{name}: ")
    getMaxGain(dset, m.attributes)

## Part 5

def buildtree(dataset, remaining_attr, level):

    if level == 2:
        return dtree.TreeLeaf(dtree.mostCommon(dataset))

    max_attr, _ = getMaxGain(dataset, remaining_attr)
    branches_dict = dict([ (value, dtree.select(dataset, max_attr, value)) for value in max_attr.values ])
    _remaining_attr = [ a for a in remaining_attr if a != max_attr ]

    branches_nodes = {}
    print (max_attr)
    for value, branch_data in branches_dict.items():
        branches_nodes[value] = buildtree(branch_data, _remaining_attr, level+1)
        
    return dtree.TreeNode(max_attr, branches_nodes, dtree.TreeLeaf(dtree.mostCommon(dataset)))

# t = buildtree(m.monk1, m.attributes, 0)

# drawtree_qt5.drawTree(t)

t = dtree.buildTree(m.monk2, m.attributes)

drawtree_qt5.drawTree(t)

# Part 5:
# for name, dset in datasets.items():
#     t = dtree.buildTree(dset, m.attributes)

#     print(name, "train", round(1 - dtree.check(t, dset), 6), end=" ")
#     print(name, "test", round(1 - dtree.check(t, testsets[name]), 6))
# exit()

## PART 6
import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
runs_per_fraction = 500

import numpy as np
import matplotlib.pyplot as plt

for name, dset in datasets.items():

    test_set = testsets[name]

    error_means = []
    error_stds = []

    for fraction in fractions:

        classification_errors = []

        for _ in range(runs_per_fraction):
            train_set, validation_set = partition(dset, fraction)

            tree = dtree.buildTree(train_set, m.attributes)
            
            while True:
                ## Repeatedly prune tree until pruned score < original score
                score = dtree.check(tree, validation_set)
                pruned = dtree.allPruned(tree)
                if len(pruned) == 0:
                    break
                pruned_results = [ (t, dtree.check(t, validation_set)) for t in pruned ]
                best_pruned_tree, best_pruned_score = max(pruned_results, key=lambda x: x[1])
                # print ("Best pruned:", score, best_pruned_score)
                if best_pruned_score <= score:
                    break
                else:
                    tree = best_pruned_tree
                
            error = 1 - dtree.check(tree, test_set)

            classification_errors.append(error)
        
        error_means.append(np.mean(classification_errors))
        error_stds.append(np.std(classification_errors))
    
    plt.xlabel("Fraction")
    plt.ylabel("Error (L1)")
    plt.title(f"Error vs Fraction Parameter ({name})")
    plt.plot(fractions, error_means, label=f"mean ({runs_per_fraction} runs)")
    plt.plot(fractions, np.array(error_means) + np.array(error_stds), "--", label=f"std ({runs_per_fraction} runs)")
    plt.plot(fractions, np.array(error_means) - np.array(error_stds), "--", label=f"std ({runs_per_fraction} runs)")
    plt.legend()
    plt.savefig(f"part6{name}.png")
    plt.show()
    plt.clf()
    print ("Means:", error_means)
    print ("Stds:", error_stds)