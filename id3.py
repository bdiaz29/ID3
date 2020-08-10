import pandas as pd
import numpy as np
import math
import collections


def cutom_log(num, base):
    if num == 0:
        return 0
    return math.log(num, base)


def entropy(count, total):
    entropy_out = (-(count / total) * cutom_log((count / total), 2)) + (
            -(count / total) * cutom_log((count / total), 2))
    return entropy_out


# return entropy of binary list
def entropy_overall(arr):
    count_0 = list(arr).count(0)
    count_1 = list(arr).count(1)
    total = len(arr)

    # entropy for out put
    entropy_out = (-(count_0 / total) * cutom_log((count_0 / total), 2)) + (
            -(count_1 / total) * cutom_log((count_1 / total), 2))
    return entropy_out


def entropy_attrb(arr, arr_2):
    count_0 = 0
    count_1 = 0

    count_0_total = 0
    count_1_total = 0

    length = len(arr_2)
    # count the 1s and 0s
    for i in range(length):
        if arr[i] == 0 and arr_2[i] == 1:
            count_0 += 1
            count_0_total += 1
        if arr[i] == 1 and arr_2[i] == 1:
            count_1 += 1
            count_1_total += 1

        if arr[i] == 0 and arr_2[i] == 0:
            count_0_total += 1
        if arr[i] == 1 and arr_2[i] == 0:
            count_1 += 1
            count_1_total += 1

    entropy_0 = entropy(count_0, count_0_total)
    entropy_1 = entropy(count_1, count_1_total)

    # take the weighted average
    entropy_avg = ((count_0 / length) * entropy_0) + ((count_1 / length) * entropy_1)
    return entropy_avg


# takes in transpose data and index of attribute to split off of
def split_data(attr_index, data, attrb_list):
    attr_top = list(attrb_list) + [0]
    attr_top.pop(attr_index)
    data_base = []
    data_0 = [np.array(attr_top)]
    data_1 = [np.array(attr_top)]
    attr_arr = data[attr_index]
    # create base by leaving out chosen atribute index
    for i in range(len(data)):
        if i == attr_index:
            continue
        data_base += [data[i]]
    data_t = np.transpose(data_base)

    # split into the two data parts
    for j in range(len(data_t)):
        temp = data_t[j]
        out = attr_arr[j]
        if out == 0:
            data_0 += [temp]
        if out == 1:
            data_1 += [temp]
        p = 0

    return data_0, data_1


def zero_or_equal(data, attributesRemaining):
    global most
    if len(data) == 0:
        return True
    data_T = np.transpose


def extract_tree(root, vals=[]):
    global placeholder
    temp = root
    if not isinstance(root, list):
        placeholder += [[vals, root]]
        return [root]
    root_0 = root[1]
    root_1 = root[2]
    base = root[0]
    place_0 = list(vals)
    place_1 = list(vals)
    place_0.extend([[base, 0]])
    place_1.extend([[base, 1]])

    print_0 = extract_tree(root_0, place_0)
    print_1 = extract_tree(root_1, place_1)
    return_list = [base, print_0, print_1]
    return return_list


def predict(root, arr):
    if not isinstance(root[1], list):
        # attributes start from 1 so subtract 1
        index = root[0] - 1
        choice = arr[index]
        if choice == 0:
            value = root[1]
        else:
            value = root[2]
        return value
    index = root[0] - 1
    choice = arr[index]
    if choice == 0:
        root_next = root[1]
    else:
        root_next = root[2]
    R = predict(root_next, arr)
    return R


# measures accuray of decision tree
def accuracy(root, data):
    hits = 0
    total = len(data)
    for i in range(len(data)):
        temp = data[i]
        x = temp[0:6]
        y = temp[6]
        pred = predict(root, temp)
        if pred == y:
            hits += 1
    accu = (hits / total) * 100
    return accu


def ID3(data, attributesRemaining, root=None):
    global most
    # if no more data remaining
    if len(data) == 0:
        return most
    # determine if leaf node
    # prepare data
    attrb_list = data[0][0:len(data[0]) - 1]
    data_list = data[1:len(data)]
    transpose_data_list = np.transpose(data_list)
    outputs = transpose_data_list[len(transpose_data_list) - 1]
    # determin if only one class label remains
    summ = np.sum(outputs)
    if summ == 0 or summ == (len(outputs)):
        return outputs[0]
    # if data cannot be split any further
    if attributesRemaining == 1:
        summ = np.sum(outputs)
        siz = len(outputs)
        if summ <= (int(siz / 2)):
            p = 0
            return 0
        else:
            p = 0
            return 1

    parent_entropy = entropy_overall(outputs)
    ent_attrb = []
    # calculate all entropies
    for i in range(attributesRemaining):
        temp = parent_entropy - entropy_attrb(transpose_data_list[i], outputs)
        ent_attrb += [temp]

    # if al entropies are equal
    np_ent_attrb = np.array(ent_attrb)
    all_equal = np.all(np_ent_attrb == np_ent_attrb[0])
    if all_equal:
        return most
    max_index = np.argmax(ent_attrb)
    max_attrb = attrb_list[max_index]
    attr_remain = attributesRemaining - 1
    # split data from chosen attribute
    data_0, data_1 = split_data(max_index, transpose_data_list, attrb_list)


    part_0 = ID3(data_0, attr_remain)
    part_1 = ID3(data_1, attr_remain)
    root = [max_attrb, part_0, part_1]
    return root


count = 0
col_names = ['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'out']

id3_tr = pd.read_table("id3-train.dat", delim_whitespace=True,
                       names=col_names,
                       header=0)
id3_train_num = id3_tr.to_numpy()
id3_train = [np.array([1, 2, 3, 4, 5, 6, 0])]
id3_train.extend(id3_train_num)

id3_train_num_transpose = np.transpose(id3_train_num)
#
## calculate entropy for entire dataset
# out_arr = id3_train_num_transpose[6]
# entropy_output = entropy_overall(out_arr)
## collected attributes
# attrbs = id3_train_num_transpose
# get num of attributres
num_attr = len(id3_train[0]) - 1

# get most frequent class from traning set
temp_0 = id3_train_num_transpose[num_attr]
sum = np.sum(temp_0)
siz = len(temp_0)
if sum <= (int(siz / 2)):
    most = 0
else:
    most = 1
# get the tree
root = ID3(id3_train, num_attr)
placeholder = []
p = extract_tree(root)

accuracy_training = accuracy(root, id3_train_num)

col_names = ['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'out']

id3_ts = pd.read_table("id3-test.dat", delim_whitespace=True,
                       names=col_names,
                       header=0)
id3_test_num = id3_ts.to_numpy()

accuracy_testing = accuracy(root, id3_test_num)

# determin depth
depth = len(placeholder[0][0])
width = len(placeholder)

str_arr = []
for o in range(int(width)):
    a_0 = placeholder[o][0][0][0]
    b_0 = placeholder[o][0][1][0]
    c_0 = placeholder[o][0][2][0]
    d_0 = placeholder[o][0][3][0]
    a_1 = placeholder[o][0][0][1]
    b_1 = placeholder[o][0][1][1]
    c_1 = placeholder[o][0][2][1]
    d_1 = placeholder[o][0][3][1]
    d_2 = placeholder[o][0][3][1]
    d_3 = placeholder[o][1]

    print("attr"+str(a_0)+" = "+str(a_1)+":")
    print("     attr"+str(b_0)+" = "+str(b_1)+":")
    print("         attr"+str(c_0)+" = "+str(c_1)+":")
    print("             attr"+str(d_0)+" = "+str(d_1)+":"+str(d_3))
    p = 0

print("training accuracy", accuracy_training, "%")
print("testing accuracy", accuracy_testing, "%")
