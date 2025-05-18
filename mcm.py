import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
import sys
import pandas as pd
from models import *
from test_sym import *
from voc_sym import *
from utils import *
import pandas as pd
import copy

torch.set_printoptions(sci_mode=False)


parser = argparse.ArgumentParser('')
parser.add_argument('--bs', type=int, default=16, help="batch size")
parser.add_argument('--nc', type=int, default=20, help="num_classes")
parser.add_argument("--nepochs", type=int, default=30, help="max epochs")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--dataset", type=str, default='voc2007')
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--root", type=str, default='/lustre/fs1/home/')
parser.add_argument("--out", type=str, default='/home/')
parser.add_argument("--noise_rate", type=float, default=0.5)
parser.add_argument("--l1_lam", type=float, default=0.001)
parser.add_argument('--p_norm', type=float, default=0.2, help="p value for L-p norm regularization")
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--lr', type=float, default=5e-6, metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--image_size', type=int, help='image_size', default=224)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))


torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)


os.makedirs(os.path.join(args.out, 'saved-models'), exist_ok=True)
os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)



out_dir = args.out+'logs'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
    run += 1
    current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
log_file = open('{}/log.txt'.format(current_dir), 'a')
print(args, file=log_file)

__console__=sys.stdout
sys.stdout=log_file




train_transform = transforms.Compose([
            MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
val_transform= transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
            
            
if(args.dataset=='voc2007'):
    
    train_dataset = Voc2007Classification(args.root,
                             set_name='train', 
                             noise_type=args.noise_type, 
                             noise_rate=args.noise_rate, 
                             transform=train_transform,
                             split_per=args.split_percentage,
                             random_seed=args.seed)

    val_dataset = Voc2007Classification(args.root,
                             set_name='val', 
                             noise_type=args.noise_type, 
                             noise_rate=args.noise_rate, 
                             transform=val_transform,
                             split_per=args.split_percentage,
                             random_seed=args.seed)

    test_dataset = Voc2007Classification(args.root, 
                            set_name='test',
                            transform=val_transform)

    
elif(args.dataset=='voc2012'):
    train_dataset = Voc2012Classification(args.root, 'train',noise_rate=args.noise_rate,transform=train_transform,random_seed=args.seed)
    val_dataset = Voc2012Classification(args.root, 'val',noise_rate=args.noise_rate,transform=val_transform,random_seed=args.seed)
    test_dataset = Voc2007Classification(args.root, 'test',transform=val_transform,random_seed=args.seed)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                          shuffle=True, num_workers=args.nworkers,drop_last=True)
                          
estimate_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                          shuffle=False, num_workers=args.nworkers,drop_last=False)                          
args.true_train_labels = train_dataset.true_labels

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs,
                        shuffle=False, num_workers=args.nworkers,drop_last=False)
                        
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs,
                        shuffle=False, num_workers=args.nworkers,drop_last=False)

    
def loss_bce_ls(preds, observed_labels, ls_coef=0.6, train_set_variant='clean'):
    assert not torch.any(observed_labels == -1), "Observed labels contain -1, which is invalid."
    assert train_set_variant == 'clean', "This loss function only supports the 'clean' variant of the training set."   
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - ls_coef) * neg_log(preds[observed_labels == 1]) + ls_coef * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - ls_coef) * neg_log(1.0 - preds[observed_labels == 0]) + ls_coef * neg_log(preds[observed_labels == 0])
    
    return loss_mtx.mean()
    
def neg_log(x):
    return - torch.log(x + 1e-5)    
        
    
class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='A', param=nn.parameter.Parameter(torch.ones(num_classes, num_classes)))
        self.register_parameter(name='B', param=nn.parameter.Parameter(torch.zeros(num_classes, num_classes)- 2)) 
        self.init= init
        self.A.to(device)
        self.B.to(device)

        self.identity = torch.eye(num_classes).to(device)

    def forward(self):
        
        
        concat_AB = torch.cat((self.A, self.B), dim=1)            
        soft_modified = concat_AB.clone() 
        soft_modified[:, :20] += self.init * self.identity

        T_C = F.softmax(soft_modified, dim=1)

        T_A, T_B = torch.split(T_C, self.A.shape[1], dim=1)

        return T_A, T_B
        
if __name__ == '__main__':

    print('seed',args.seed)
    net = get_resnet50(args.nc,pretrained=True)    
    net=net.to(device)
        

    if args.dataset =='voc2007':
        init = 5.75
    elif args.dataset =='voc2012':
        init = 5.75
    else:
        init= 6.43
    
    trans = sig_t(device, args.nc, init=init)
    trans.cuda()
    net_lr=args.lr


    optimizer = torch.optim.Adam(
        [
            {'params': net.parameters(), 'lr': net_lr/10},
            {'params': trans.parameters(), 'lr': net_lr*10}
        ], 
        weight_decay=args.weight_decay
    )
  
    
    val_metric=[]
    test_metric=[]
    
    best_a = torch.empty(args.nc, args.nc)
    best_b = torch.empty(args.nc, args.nc)
    
     
    best_val=0
    for i in range(args.nepochs):
        start = time.time()
        
        train_total = 0
        train_loss = 0.0
        
        for idx, (data, labels) in enumerate(train_loader):
            data = data.cuda().float()
            labels = labels.cuda().float()
            batch_size = data.size(0)
            train_total += 1
            f = net(data)   
            f = torch.sigmoid(f)
            l_regularization = args.l1_lam * torch.sum(torch.pow(torch.norm(f, dim=0, p=args.p_norm), args.p_norm)) / batch_size
            A, B = trans()
            Af = torch.matmul(A, f.T).T
            B1_f = torch.matmul(B, (1 - f).T).T
            g_n = Af + B1_f
            loss = loss_bce_ls(g_n, labels, ls_coef=0.1)        

            loss += l_regularization
            loss.backward()               
            
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss
        train_loss_ = train_loss / float(train_total)
        
        
        val_1, val_2 = test(net, val_loader, return_map=True)
        val_loss, v_hloss, v_rloss, cover, avgpre, oneerror, acc = val_1
        map, OP, OR, OF1, CP, CR, CF1 = val_2
        
        print('Epoch',i,' val_map, OP, OR, OF1, CP, CR, CF1 ',round(map,2), round(OP,2), round(OR,2), round(OF1,2), round(CP,2),round(CR,2),round(CF1,2),round(oneerror,2))  
        val_metric.append([map, OP, OR, OF1, CP, CR, CF1, oneerror])
        if(map>best_val):
            best_val=map
            best_a = A 
            best_b = B
            best_state_dict = copy.deepcopy(net.state_dict())
        
        test_1, test_2 = test(net, test_loader, return_map=True)        
        val_loss, v_hloss, v_rloss, cover, avgpre, oneerror, acc = test_1
        map, OP, OR, OF1, CP, CR, CF1 = test_2
        
        print('Epoch',i,'test_map, OP, OR, OF1, CP, CR, CF1, oneerror ',round(map,2), round(OP,2), round(OR,2), round(OF1,2), round(CP,2),round(CR,2),round(CF1,2),round(oneerror,2))  
        test_metric.append([map, OP, OR, OF1, CP, CR, CF1, oneerror])
        end = time.time()
        print('time', round(end-start,3))
        log_file.flush()

    net.load_state_dict(best_state_dict)

    val_metric = np.array(val_metric)
    test_metric = np.array(test_metric)
    

    best_map_idx, _, _, best_OF1_idx, _, _, best_CF1_idx, _ = np.argmax(val_metric, axis=0)
    

    print('Best validation results (based on mAP):')
    print(' val_map, OP, OR, OF1, CP, CR, CF1 =', 
          round(val_metric[best_map_idx, 0], 2), 
          round(val_metric[best_map_idx, 1], 2), 
          round(val_metric[best_map_idx, 2], 2), 
          round(val_metric[best_map_idx, 3], 2), 
          round(val_metric[best_map_idx, 4], 2), 
          round(val_metric[best_map_idx, 5], 2), 
          round(val_metric[best_map_idx, 6], 2))
    

    print('Test results corresponding to best val_map:')
    print(' test_map, OP, OR, OF1, CP, CR, CF1 =', 
          round(test_metric[best_map_idx, 0], 2), 
          round(test_metric[best_map_idx, 1], 2), 
          round(test_metric[best_map_idx, 2], 2), 
          round(test_metric[best_map_idx, 3], 2), 
          round(test_metric[best_map_idx, 4], 2), 
          round(test_metric[best_map_idx, 5], 2), 
          round(test_metric[best_map_idx, 6], 2))
    
    log_file.flush()