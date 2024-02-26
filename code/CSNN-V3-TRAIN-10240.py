import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
import numpy as np

import torch, gc

gc.collect()
torch.cuda.empty_cache()

# 网络结构
class SNN_VGG9(nn.Module):
    def __init__(self, T: int, use_cupy=False):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(

            # Conv Block 1
            layer.Conv2d(1, 64, kernel_size=(3, 2), stride=1, padding='same'),  # (64, 81920, 2)
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(64, 64, kernel_size=(3, 2), stride=1, padding='same'),  # (64, 81920, 2)
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(kernel_size=(4, 2), stride=(4, 2)),  # (64, 20480, 1)

            # Conv Block 2
            layer.Conv2d(64, 128, kernel_size=(3, 1), stride=1, padding='same'),  # (128, 20480, 1)
            layer.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(128, 128, kernel_size=(3, 1), stride=1, padding='same'),  # (128, 20480, 1)
            layer.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(kernel_size=(4, 1), stride=(4, 1)),  # (128, 5120, 1)

            # Conv Block 3
            layer.Conv2d(128, 256, kernel_size=(3, 1), stride=1, padding='same'),  # (256, 5120, 1)
            layer.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding='same'),  # (256, 5120, 1)
            layer.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding='same'),  # (256, 5120, 1)
            layer.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(kernel_size=(8, 1), stride=(8, 1)),  # (256, 640, 1)

            layer.Flatten(),
            layer.Linear(20480, 640, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(640, 12, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)

        fr = x_seq.mean(0)
        return fr

# 参数设置、训练和测试
def main():
    parser = argparse.ArgumentParser(description='Classify 007-Data-system')
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-amp', default=False, action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', default=True, action='store_true', help='use cupy backend')
    parser.add_argument('-opt', default='adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')

    args = parser.parse_args()
    print(args)

    net = SNN_VGG9(T=args.T, use_cupy=args.cupy)

    print(net)

    net.to(args.device)

    # 加载数据
    path = r"autodl-tmp/protocol12/data/"
    x_train, x_test, y_train, y_test = np.load(path + 'x_train.npy'), np.load(path + 'x_test.npy'), np.load(
        path + 'y_train.npy'), np.load(path + 'y_test.npy')
    # resize为(x_train.shape[0], 1, 1024, 2)
    # 那么一维代表：样本数，1代表： ，1024*2代表：IQ数组
    x_train = np.resize(x_train, (x_train.shape[0], 1, 10240, 2))
    x_test = np.resize(x_test, (x_test.shape[0], 1, 10240, 2))

    print("X_train.shape", x_train.shape)
    print("Y_train.shape", y_train.shape)
    print("X_test.shape", x_test.shape)
    print("Y_test.shape", y_test.shape)
    # torch.from_numpy把数组转换成张量,且二者共享内存
    x_train, x_test, y_train, y_test = torch.from_numpy(x_train), torch.from_numpy(x_test), torch.from_numpy(
        y_train), torch.from_numpy(y_test)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        num_workers=args.j,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        num_workers=args.j,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    out_dir = os.path.join(args.out_dir,
                           f'007-data-protocol12-10240-SCNN-VGG9-T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label_onehot in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device).to(torch.float32)

            label_onehot = label_onehot.to(args.device).to(torch.float32)
            label = label_onehot.argmax(1)
            # print(label)
            # print(label.numel())

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label_onehot in test_data_loader:
                img = img.to(args.device).to(torch.float32)

                label_onehot = label_onehot.to(args.device).to(torch.float32)

                label = label_onehot.argmax(1)

                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
            print("-----The best model until now has been saved-----")

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')

        epoch_time = (test_time - start_time) / 60
        print(f'train time consuming ={epoch_time: .4f} minute')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
