## Assignment 0
_Each one of the datasets has properties which makes them hard to learn. Motivate which of the three problems is most difficult for a decision tree algorithm to learn._

Problem 3's underlying truth will be difficult for the decision tree to learn, as there is noise in the training set that will cause overfitting. Problem 1 and 2 has no noise, so what the tree learns is completely representative.

## Assignment 1
_The file dtree.py defines a function entropy which calculates the entropy of a dataset. Import this file along with the monks datasets and use it to calculate the entropy of the training datasets._

```
Entropy(MONK-1) = 1.0
Entropy(MONK-2) = 0.957117428264771
Entropy(MONK-3) = 0.9998061328047111
```

*Notes: MONK-2 training set is biased towards positive or negative examples and is not a uniform distribution.*

## Assignment 2
_Explain entropy for a uniform distribution and a non-uniform distribution, present some example distributions with high and low entropy._

For discrete entropy with a fixed number of outcomes N, a uniform distribution will have entropy log<sub>2</sub>(N).
The uniform distribution will always have more entropy than a non-uniform distribution because with a uniform distribution, we have no knowledge/bias of which state will be chosen (all of them are equally likely). With a non-uniform distribution, we know which state will be chosen more frequently.

Given only two possible outcomes s0, s1.\
A distribution where\
P(s0) = P(s1) = 0.5 (Ex. A fair coin toss)\
will have the maximum entropy while a distribution where\
P(s0) = 0, P(s1) = 1 (Ex. Asking if the coin will land on head OR tail)\
will have the lowest entropy of 0

## Assignment 3
_Use the function averageGain (defined in dtree.py) to calculate the expected information gain corresponding to each of the six attributes. Note that the attributes are represented as in- stances of the class Attribute (defined in monkdata.py) which you can access via m.attributes[0], ..., m.attributes[5]. Based on the results, which attribute should be used for splitting the examples at the root node?_

```
MONK-1:
[('A1', 0.07527255560831925), ('A2', 0.005838429962909286), ('A3', 0.00470756661729721), ('A4', 0.02631169650768228), ('A5', 0.28703074971578435), ('A6', 0.0007578557158638421)]
Max gain on attribute ('A5', 0.28703074971578435)
MONK-2:
[('A1', 0.0037561773775118823), ('A2', 0.0024584986660830532), ('A3', 0.0010561477158920196), ('A4', 0.015664247292643818), ('A5', 0.01727717693791797), ('A6', 0.006247622236881467)]
Max gain on attribute ('A5', 0.01727717693791797)
MONK-3:
[('A1', 0.007120868396071844), ('A2', 0.29373617350838865), ('A3', 0.0008311140445336207), ('A4', 0.002891817288654397), ('A5', 0.25591172461972755), ('A6', 0.007077026074097326)]
Max gain on attribute ('A2', 0.29373617350838865)
```

We want to use:
A5 for MONK-1
A5 for MONK-2
A2 for MONK-3

## Assignment 4
_For splitting we choose the attribute that maximizes the information gain, Eq.3. Looking at Eq.3 how does the entropy of the subsets, Sk, look like when the information gain is maximized? How can we motivate using the information gain as a heuristic for picking an attribute for splitting? Think about reduction in entropy after the split and what the entropy implies._

Entropy implies uncertainties when choosing a label (positive and negative). Minimizing entropy is reducing the uncertainty, and maximizing the confidence. The subset Sk will have the minimum entropy when information gain is maximized.
When we choose the attribute that has the greatest information gain, the subsets after the splits will have minimum entropy. Since the goal is to reduce entropy in as few branches as possible, using the largest decrease in entropy is a greedy heuristic for reaching the goal.

## 5 Some stuff

```
Which attributes should be tested for these nodes?
```

Only attributes that have not been selected, aka the attribute that was selected at the root.

## Assignment 5
_Compute the train and test set errors for the three Monk datasets for the full trees. Were your assumptions about the datasets correct? Explain the results you get for the training and test datasets._

```
MONK-1 train 1.0
MONK-1 test 0.8287037037037037
MONK-2 train 1.0
MONK-2 test 0.6921296296296297
MONK-3 train 1.0
MONK-3 test 0.9444444444444444
```

The train dataset will always return classification that is correct, as the tree is fitted exactly to the dataset.

My initial assumption for the datasets were wrong. MONK-3 performed the best during validation while MONK-2 performed the worse.

## Assignment 6
_Explain pruning from a bias variance trade-off perspective._

Pruning prevents overfitting of the training data, which decreases variance but also increases bias. 
One can think of pruning a node, which makes some decision about the data, as reducing the complexity of the system and limiting its degree of freedom.
However, with too much pruning, the bias would increase since the system is unable to model the complexity of the underlying truth.

## Assignment 7

```
Means: [0.22407407407407404, 0.20711805555555554, 0.19814814814814813, 0.15520833333333334, 0.1509837962962963, 0.13049768518518517]
Stds: [0.04509791587713235, 0.028236886405921105, 0.04061098338970982, 0.04674249676625926, 0.04601327699255532, 0.04770621474216665]
Means: [0.33327546296296295, 0.3340856481481481, 0.33425925925925926, 0.3343171296296296, 0.3358796296296297, 0.33842592592592596]
Stds: [0.013640599945158368, 0.012512719651532441, 0.013604338198995475, 0.011881197168238074, 0.012771402059383069, 0.016263118583565078]
Means: [0.07754629629629631, 0.05266203703703705, 0.050810185185185194, 0.039120370370370375, 0.040162037037037045, 0.045486111111111116]
Stds: [0.05230405903853659, 0.039005659949025155, 0.03420114408949177, 0.02701399798448249, 0.030188371437719665, 0.02872399782879921]
```