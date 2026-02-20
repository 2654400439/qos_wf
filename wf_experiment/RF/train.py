# encoding: utf8

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import os
from RF import getRF
import const_rf as const

EPOCH = 60
BATCH_SIZE = 200
LR = 0.0005
if_use_gpu = 1
num_classes = const.num_classes

def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    return train_X, train_y


def adjust_learning_rate(optimizer, echo):
    lr = LR * (0.2 ** (echo / EPOCH))
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr


def val(cnn, test_x, test_y, result_file, test_file):
    cnn.eval()
    test_result = open(test_file, 'w+')
    test_data = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    for step, (tr_x, tr_y) in enumerate(test_loader):
        test_output = cnn(tr_x)
        if if_use_gpu:
            test_output = test_output.cpu()
        pred_y, accuracy = get_result(test_output, tr_y)
        resultfile = open(result_file, 'w+')
        for i in range(len(tr_y)):
            resultfile.write(str(tr_y[i].numpy()) + ',' + str(pred_y[i]) + '\n')
        resultfile.close()
        test_result.write(str(accuracy) + '\n')
        print(accuracy)
    test_result.close()


def get_result(output, true_y):
    pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
    accuracy = (pred_y == true_y.numpy()).sum().item() * 1.0 / float(true_y.size(0))
    return pred_y, accuracy


def control(feature_file):
    x, y = load_data(feature_file)
    cnn = getRF(num_classes)

    if if_use_gpu:
        cnn = cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.001)
    loss_func = nn.CrossEntropyLoss()
    train_x = torch.unsqueeze(torch.from_numpy(x), dim=1).type(torch.FloatTensor)
    train_x = train_x.view(train_x.size(0), 1, 2, -1)
    train_y = torch.from_numpy(y).type(torch.LongTensor)

    train_y = train_y - 1

    train_data = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    cnn.train()

    for epoch in range(EPOCH):
        adjust_learning_rate(optimizer, epoch)
        for step, (tr_x, tr_y) in enumerate(train_loader):
            batch_x = Variable(tr_x.cuda())
            batch_y = Variable(tr_y.cuda())

            # 模型前向传播
            output = cnn(batch_x)

            # ================= [开始插入 Debug 代码] =================
            # 1. 获取模型认为的类别数 (Output 的第二维)
            model_n_classes = output.shape[1]

            # 2. 获取当前 Batch 真实标签的最大值和最小值
            label_max = batch_y.max().item()
            label_min = batch_y.min().item()

            # 3. 核心检查：如果标签超出 [0, n_classes-1] 的范围，立即报错并打印详情
            if label_max >= model_n_classes or label_min < 0:
                print(f"\n{'=' * 40}")
                print(f"[CRITICAL ERROR DETECTED] Epoch {epoch}, Step {step}")
                print(f"1. Model Output Shape: {output.shape}")
                print(f"   -> Model expects labels in range [0, {model_n_classes - 1}]")
                print(f"2. Actual Batch Labels: Min={label_min}, Max={label_max}")

                if label_max >= model_n_classes:
                    print(f"   -> ERROR: Label {label_max} is too large! (>= {model_n_classes})")
                    print(f"   -> Check: Are your labels 1-based (e.g., 1-100)? PyTorch requires 0-based (0-99).")

                if label_min < 0:
                    print(f"   -> ERROR: Label {label_min} is negative!")

                print(f"{'=' * 40}\n")
                import sys;
                sys.exit(1)  # 强行停止，防止后续的 CUDA 报错掩盖真相
            # ================= [结束插入 Debug 代码] =================

            _, accuracy = get_result(output.cpu(), tr_y.cpu())

            del batch_x

            # 计算 Loss (原本出错的地方)
            loss = loss_func(output, batch_y)

            del batch_y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del output

            if step % 10 == 0:
                print(epoch, step, accuracy, loss.item())

    torch.save(cnn.state_dict(), os.path.join(const.model_path, method + '.pth'))


if __name__ == '__main__':
    # TODO: change the data file path
    defense = 'Undefended'
    # feature_file = '../countermeasure/dataset/' + defense + '.npy'
    feature_file = '../preprocessed_data/RF_features-train.npy'
    method = defense
    control(feature_file)