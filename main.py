import argparse
import random
import numpy as np
import torch
from ResNet import resnet8_cbam
from PASS import ProtoAugSSL
import torch.backends.cudnn as cudnn
from DATA import DATA
from torch.utils.data import DataLoader
from sklearn import metrics

parser = argparse.ArgumentParser(description='PASS')
parser.add_argument('--epochs_old', default=11, type=int)
parser.add_argument('--epochs_new', default=5, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--data_name', default='PaviaU', type=str)
parser.add_argument('--total_nc', default=9, type=int)
parser.add_argument('--fg_nc', default=8, type=int)
parser.add_argument('--task_num', default=1, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--new_lr', default=0.00012, type=float)
parser.add_argument('--protoAug_weight', default=0.26, type=float)
parser.add_argument('--kd_weight', default=0.29, type=float)
parser.add_argument('--temp', default=0.1, type=float)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--seed', default=96, type=int)
parser.add_argument('--save_path', default='model_saved/', type=str)

args = parser.parse_args()
print(args)


def set_seed(seed):
    if seed == 0:
        print('random seed')
        cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def main():
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '_' + str(task_size)

    feature_extractor = resnet8_cbam()
    model = ProtoAugSSL(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))

    for i in range(args.task_num + 1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.before_train(i)
        model.train(i, old_class=old_class)
        model.after_train()

    print(' ')
    print("#" * 25 + "Test for up2now Task" + "#" * 25)
    test_dataset = DATA(train=False)
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % class_index
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.get_test_data_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=test_dataset.data.shape[0])
        correct, total = 0.0, 0.0
        for step, (data, labels) in enumerate(test_loader):
            datas, labels = data.to(device), labels.to(device)
            datas = datas.view(-1, 103, 1, 1)
            with torch.no_grad():
                outputs = model(datas)
            outputs = outputs[:, ::3]

            predicts = torch.max(outputs, dim=1)[1]
            overall_accuracy = format(metrics.accuracy_score(labels.cpu(), predicts.cpu()))
            acc_for_each_class = metrics.recall_score(labels.cpu(), predicts.cpu(), average=None)
            average_accuracy = format(np.mean(acc_for_each_class))
            kappa = format(metrics.cohen_kappa_score(labels.cpu(), predicts.cpu()))

            print('acc_for_each_class:', acc_for_each_class)
            print("overall_accuracy:%.5f, average_accuracy:%.5f, kappa:%.5f" % (
                float(overall_accuracy), float(average_accuracy), float(kappa)))
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print("accuracy", accuracy)


if __name__ == '__main__':
    set_seed(args.seed)
    main()
