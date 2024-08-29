import os
import pickle
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from datasets import *
from models import *
from utils.visualize import Visualizer
from utils.utils_app import save_state, load_state_resume

from torchnet import meter
                
def test_steps(model, ce_loader_test, best_acc, best_pos, best_neg, vis, prefix):
    # eval
    model.eval()

    # iter
    with torch.no_grad():
        count0 = 0
        count1 = 0
        count2 = 0
        count0_pos = 0
        count0_neg = 0
        count2_pos = 0
        count2_neg = 0
        record = {}
        for data in tqdm(ce_loader_test):
            inputs, labels = (data[0][0].to(device), data[0][1].to(device), data[0][2].to(device)), data[1].to(device)
            pairs = data[2].numpy()
            
            output = model(inputs)
            pred_ = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            pred = (pred_ >= 0.5)
            labels = labels.cpu().numpy()

            # acc
            count0 += labels.shape[0]
            count2 += (pred==labels).sum().item()

            pos = (labels==1)
            count0_pos += pos.sum().item()
            count2_pos += (pred[pos]==labels[pos]).sum().item()

            neg = (labels!=1)
            count0_neg += neg.sum().item()
            count2_neg += (pred[neg]==labels[neg]).sum().item()
            
            # record
            for i in range(len(pairs)):
                ap = (pairs[i][0],pairs[i][1])
                record[ap] = pred_[i]

            #break

        tmp_acc = 100.0*count2/(count0+1e-5)
        tmp_pos = 100.0*count2_pos/(count0_pos+1e-5) 
        tmp_neg = 100.0*count2_neg/(count0_neg+1e-5)
        vis.plot('tmp_acc', tmp_acc)
        vis.plot('tmp_pos', tmp_pos)
        vis.plot('tmp_neg', tmp_neg)

        if tmp_acc > best_acc:
            vis.slogger.info("==== save save ====")
            best_acc = tmp_acc
            # save model
            save_state({
                'state_dict': model.state_dict(),
                }, "checkpoints/{}".format("try"), "best")
            # save record
            with open("temp/try_best.json", 'wb') as g:
                pickle.dump(record, g)
        if tmp_pos > best_pos:
            best_pos = tmp_pos
        if tmp_neg > best_neg:
            best_neg = tmp_neg
        vis.slogger.info("{} acc {:.4f}({}/{}) pos {:.4f}({}/{}) neg {:.4f}({}/{}) best acc {:.4f} pos {:.4f} neg {:.4f}".format(
            prefix,
            tmp_acc, count2, count0,
            tmp_pos, count2_pos, count0_pos,
            tmp_neg, count2_neg, count0_neg,
            best_acc, best_pos, best_neg
            ))

    # train
    model.train()
    return  best_acc, best_pos, best_neg

if __name__ == '__main__':
    # os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # arg
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256) # 512

    parser.add_argument('--k', type=int, default=80)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)

    part = "part0_train"
    part_test = "part1_test"
    parser.add_argument('--feat_path', type=str, default='../data/features/{}.bin'.format(part))
    parser.add_argument('--label_path', type=str, default='../data/labels/{}.meta'.format(part))
    parser.add_argument('--knn_graph_path', type=str, default='../data/knns/{}/knn_nbrs.npz'.format(part))

    # args
    args = parser.parse_args()

    # vis
    vis = Visualizer(env="trytoo3")

    # device
    device = torch.device('cuda')

    # dataset 
    # 0.05/0.60 0.15/0.60
    # 0.05/0.63 0.15/0.63
    ce_dataset = PCENetDataset(dataset_name=part, 
            feat_path=args.feat_path, 
            label_path=args.label_path,
            knn_graph_path=args.knn_graph_path, 
            feature_dim=args.feature_dim, k=args.k, vaa=0.22, vbb=0.12, vis=vis) # parameters for theta and delta
    ce_loader = torch.utils.data.DataLoader(ce_dataset, batch_size=args.batch_size, num_workers=4,
                             shuffle=True, drop_last=False)

    ce_dataset_test = PCENetDataset(dataset_name=part_test, 
            feat_path=args.feat_path.replace(part, part_test), 
            label_path=args.label_path.replace(part, part_test),
            knn_graph_path=args.knn_graph_path.replace(part, part_test), 
            feature_dim=args.feature_dim, k=args.k, vaa=0.22, vbb=0.12, vis=vis) # parameters for theta and delta
    ce_loader_test = torch.utils.data.DataLoader(ce_dataset_test, batch_size=args.batch_size, num_workers=4,
                             shuffle=False, drop_last=False)

    # model
    model = PCENet(feature_dim=args.feature_dim, k=args.k)
    
    #load_path = "x.pth"
    #vis.slogger.info("==== load path {}".format(load_path))
    #model.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))

    #model = torch.nn.DataParallel(model)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

    # load
    start_epoch = 0
    #start_epoch = load_state_resume("x.pth", model, optimizer)

    # loss
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)

    # meter
    loss_meter = meter.MovingAverageValueMeter(100)

    # train
    epoch_max = 8
    model.train()
    best_acc = 0
    best_pos = 0
    best_neg = 0
    for epoch in range(start_epoch,epoch_max):
        vis.slogger.info("==== epoch [{}/{}]".format(epoch+1, epoch_max))
        count = 0
        count0 = 0
        count1 = 0
        count2 = 0
        count0_pos = 0
        count0_neg = 0
        count2_pos = 0
        count2_neg = 0
        for data in (ce_loader):
            inputs, labels = (data[0][0].to(device), data[0][1].to(device), data[0][2].to(device)), data[1].to(device)
            count0 += labels.shape[0]
            #count1 += (labels==1).sum().item()
            #print(count1, count0, 100.0*count1/count0)
            #continue

            output = model(inputs)
            loss = loss_func(output, labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss meter
            loss_meter.add(loss.item())

            # acc
            pred = F.softmax(output, dim=1)[:, 1]
            pred = pred >= 0.5
            count2 += (pred==labels).sum().item()

            pos = (labels==1)
            count0_pos += pos.sum().item()
            count2_pos += (pred[pos]==labels[pos]).sum().item()

            neg = (labels!=1)
            count0_neg += neg.sum().item()
            count2_neg += (pred[neg]==labels[neg]).sum().item()

            # plot
            if (count+1) % 50 == 0: # 20
                vis.slogger.info("==== epoch [{}/{}] iter [{}/{}] loss {:.4f}({:.4f}) acc {:.4f}({}/{}) pos {:.4f}({}/{}) neg {:.4f}({}/{})".format(
                    epoch+1, epoch_max, 
                    count+1, len(ce_loader),
                    loss.item(), loss_meter.value()[0],
                    100.0*count2/(count0+1e-5), count2, count0,
                    100.0*count2_pos/(count0_pos+1e-5), count2_pos, count0_pos,
                    100.0*count2_neg/(count0_neg+1e-5), count2_neg, count0_neg,
                    ))
            
            if (count+1) % 1000 == 0:
                vis.plot('loss', loss_meter.value()[0])
                vis.plot('acc', 100.0*count2/(count0+1e-5))
                vis.plot('pos', 100.0*count2_pos/(count0_pos+1e-5))
                vis.plot('neg', 100.0*count2_neg/(count0_neg+1e-5))
            
            if (count+1) % 14000 == 0:
                # test
                prefix = "==== epoch [{}/{}] iter [{}/{}]".format(
                    epoch+1, epoch_max, 
                    count+1, len(ce_loader))
                best_acc, best_pos, best_neg = test_steps(model, ce_loader_test, best_acc, best_pos, best_neg, vis, prefix)

            count += 1
            #break

        # save
        save_state({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, 'checkpoints/{}'.format("try"), epoch+1)
