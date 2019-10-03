import monkdata as m
import itertools

# print (m.attributes)

# total = 0
# for comb in itertools.combinations(range(6), 2):
#     remain = [i for i in range(6) if i not in comb]
    
#     mul = 1

#     for r in remain:
#         mul *= len(m.attributes[r].values) - 1
    
#     total += mul
#     print (total, comb, [len(m.attributes[r].values) for r in remain])

# print (total)

A1 = m.attributes[0]
A2 = m.attributes[1]
A3 = m.attributes[2]
A4 = m.attributes[3]
A5 = m.attributes[4]
A6 = m.attributes[5]

count1 = 0
count2 = 0
count3 = 0
for i in range(len(m.monk1test)):
    attr = m.monk1test[i].attribute
    if attr[A1] == attr[A2] or attr[A5] == 1:
        print (attr)
        count1 += 1

print (count1)