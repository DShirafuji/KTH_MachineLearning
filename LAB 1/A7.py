import dtree as d
import monkdata as m
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from statistics import variance


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def Average(lst):
    return sum(lst) / len(lst)


fraction_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90]
#fraction_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

accuracy1 = 100
accuracy2 = 100
accuracy3 = 100


accuracy_list_1 = []
accuracy_list_2 = []
accuracy_list_3 = []

vari_list_1 = []
vari_list_2 = []
vari_list_3 = []

# delete same elements with
def delete_same_elements(monk, monk_test):
    id_list = []
    for each_m in monk:
        id_list.append(each_m.identity)
    test =[]
    for each_m_test in monk_test:
        if (each_m_test.identity in id_list):
            continue
        else:
            test.append(each_m_test)
    tu_test = tuple(test)
    return(tu_test)



monk_1 = m.monk1
monk_1_test = m.monk1test
m1_test = delete_same_elements(monk_1, monk_1_test)

monk_2 = m.monk2
monk_2_test = m.monk2test
m2_test = delete_same_elements(monk_2, monk_2_test)

monk_3 = m.monk3
monk_3_test = m.monk3test
m3_test = delete_same_elements(monk_3, monk_3_test)

# We need to optimize the fraction
for fraction in fraction_list:
    list1 = []
    list2 = []
    list3 = []
    for loop in range(0,100):
        monk1train, monk1val = partition(m.monk1, fraction)
        t=d.buildTree(monk1train, m.attributes)
        list1.append(1-d.check(t, monk_1_test))

        monk2train, monk2val = partition(m.monk2, fraction)
        t=d.buildTree(monk2train, m.attributes)
        list2.append(1-d.check(t, monk_2_test))

        monk3train, monk3val = partition(m.monk3, fraction)
        t=d.buildTree(monk3train, m.attributes)
        list3.append(1-d.check(t, monk_3_test))

    temp1 = Average(list1)
    temp2 = Average(list2)
    temp3 = Average(list3)

    vari_list_1.append(list1)
    vari_list_2.append(list2)
    vari_list_3.append(list3)

    accuracy_list_1.append(temp1)
    accuracy_list_2.append(temp2)
    accuracy_list_3.append(temp3)
    
    if temp1 < accuracy1:
        good_fraction1 = fraction
        accuracy1 = temp1
    if temp2 < accuracy2:
        good_fraction2 = fraction
        accuracy2 = temp2
    if temp3 < accuracy3:
        good_fraction3 = fraction
        accuracy3 = temp3

V1 = []
V2 = []
V3 = []
for num_len in range(len(vari_list_1)):
    V1.append(variance(vari_list_1[num_len]))
    V2.append(variance(vari_list_2[num_len]))
    V3.append(variance(vari_list_3[num_len]))

print("=== Test ===")
print(str(good_fraction1) + " : " + str(accuracy1))
print(str(good_fraction2) + " : " + str(accuracy2))
print(str(good_fraction3) + " : " + str(accuracy3))

print("V1:")
print(V1)
print("V2:")
print(V2)
print("V3:")
print(V3)

plt.errorbar(fraction_list, accuracy_list_1, V1, label="MONK-1", marker='None')
plt.errorbar(fraction_list, accuracy_list_2, V2, label="MONK-3", marker='None')
plt.errorbar(fraction_list, accuracy_list_3, V3, label="MONK-3", marker='None')
plt.show()
