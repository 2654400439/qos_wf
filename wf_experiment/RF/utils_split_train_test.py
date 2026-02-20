import random

train_list = []
test_list = []

for i in range(1, 1174):
    index = list(range(1, 13))
    random.shuffle(index)

    train = index[:8]
    test = index[10:]
    for item in train:
        train_list.append(str(i)+'-'+str(item)+'.txt')
    for item in test:
        test_list.append(str(i)+'-'+str(item)+'.txt')

with open('./list/Index_train.txt', 'w') as f:
    for item in train_list:
        f.write(item+'\n')

with open('./list/Index_test.txt', 'w') as f:
    for item in test_list:
        f.write(item+'\n')