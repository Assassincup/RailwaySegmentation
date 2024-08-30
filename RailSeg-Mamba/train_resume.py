import argparse
#from data.preprocess.repc.repc_dataset_sparse import build_dataloader
from data.preprocess.repc.repc_dataset import build_dataloader
# from data.preprocess.repc.repc_dataset_randla import build_dataloader
# from data.preprocess.repc.repc_dataset_ptv3 import build_dataloader
from model.PCT.pct import down_sample
from model.model_utils import load_data_to_device
from optim.opt_and_sche import build_optimizer, build_scheduler
import torch
import numpy as np #版本一：2.0.1 版本二1.25.0
from tqdm import tqdm
import os
import torch.distributed as dist
from collections import OrderedDict
import time
import logging
import gc
# 用PointTransformerV3的时候将其余的方法注释，之后需要将这个部分优化弄成可以选择的配置文件
# from model.PCT.PCT_TV import train_pct,val_pct
# from model.PCT.pct import PCSeg
# from model.SPVCNN.SPVCNN_TV import train_spvcnn,val_spvcnn
# from model.SPVCNN.spvcnn import SPVCNN
# from model.POINT_MAMBA.POINT_MAMBA_TV import train_point_mamba,val_point_mamba
# from model.POINT_MAMBA.point_mamba import PointMamba
from model.U_MAMBA.U_MAMBA_TV import train_u_mamba,val_u_mamba
# #from model.U_MAMBA.u_mamba import U_MAMBA
# #from model.U_MAMBA.u_mamba_spv import U_MAMBA
# from model.U_MAMBA.mamba2order import realization
from model.U_MAMBA.RailRealizationPointMamba import U_MAMBA
# from model.PointTransformerV3.PTV3 import PointTransformerV3
# from model.PointTransformerV3.PTV3_TV import  train_point_transformerv3,val_point_transformerv3
# from model.RANDLANET.RandLANet import Network as RandLANet
# from model.RANDLANET.RANDLANET_TV import train_randla,val_randla
# from data.preprocess.repc.utils_randla.config import ConfigSemanticKITTI as randla_config
# from model.POINTNET.pointnet import POINTNET
# from model.POINTNET.pointnet2 import POINTNET2
# from model.POINTNET.POINTNET_TV import train_pointnet,val_pointnet
from matplotlib import pyplot as plt
gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
best_mean_ious = 0
time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.basicConfig(level=logging.INFO,filename=f'log//train_{time}.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OPT():
    def __init__(self):
        self.OPTIMIZER = "adam"
        self.LR = 0.001#0.002
        self.LR_PER_SAMPLE= 0.001 #0.02
        self.WEIGHT_DECAY=0.0001
        self.MOMENTUM=0.9
        self.NESTEROV=True
        self.GRAD_NORM_CLIP=10
        self.ARMUP_EPOCH = 1
        self.WARMUP_EPOCH = 2
        self.SCHEDULER = "linear_warmup_with_cosdecay"
class cfgs():
    def __init__(self):
        self.class_num = 11
        self.optimizer = OPT()
        self.epochs = 400

def get_cfgs():
    my_cfgs = cfgs()
    return my_cfgs

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("-l","--local-rank",type=int,default=-1,help="GPU rank")
    parse.add_argument("-d","--dataset",type=str,default="RepcDataset",help="the dataset you want to use")
    parse.add_argument("-m", "--mode" , type=str , default = "train" , help="train or tset")
    parse.add_argument('-r',"--resume",action='store_true')
    parse.add_argument('-e',"--start_epoch",type=int,default=0,help="choose your start epoch")
    #print("parse.parse_args().resume",parse.parse_args().resume)
    parse.add_argument('-p',"--checkpoint_path",type=str,default="./checkpoints/checkpoint_last_2.pth",help="choose your checkpoint")
    parse.add_argument('-s',"--model",type=str,default="POINTNET",help="select your model")
    return parse.parse_args()

def IOU(preds, labels, part_num):
    ious = []
    print("part_num",part_num)
    for label in range(part_num):
        pred_mask = preds == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).sum() / (pred_mask | labels_mask).sum()
        ious.append(iou)
    ious.append(np.nanmean(ious[1:]))
    print("len:ious",len(ious))
    return ious


class Trainer():
    def __init__(self,args,cfgs):
        self.args = args
        self.cfgs = cfgs

        dataset, dataloader, sampler = build_dataloader(mode = self.args.mode)
        self.dataset = dataset
        self.dataloader = dataloader
        self.sampler = sampler
        self.start_epoch = -1
        self.modelname = args.model
        self.class_name = ['void', 'tree', 'line', 'pole', 'building', 'rail', 'pathway', 'bar', 'hillside', 'tunnel',
                  'platform', 'mean']

        if self.args.local_rank != -1:
            assert torch.cuda.device_count() > self.args.local_rank
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device('cuda',self.args.local_rank)
        else:
            self.device = torch.device('cpu')

        self.rank = 0 if self.args.local_rank < 0 else dist.get_rank()

        #model = PCSeg(part_num = self.cfgs.class_num)
        #model = SPVCNN(class_num = self.cfgs.class_num)
        if self.modelname=="Pc_Seg":
            model = PCSeg(part_num=self.cfgs.class_num)
            self.train_model = train_pct
            self.val_model = val_pct
        elif self.modelname == "SPVCNN":
            model = SPVCNN(class_num=self.cfgs.class_num)
            self.train_model = train_spvcnn
            self.val_model = val_spvcnn
        elif self.modelname == "POINT_MAMBA":
            model = PointMamba(class_num=self.cfgs.class_num)
            self.train_model = train_point_mamba
            self.val_model = val_point_mamba
        elif self.modelname == "U_MAMBA" or self.modelname == "U_MAMBA_index":
            model = U_MAMBA(class_num=self.cfgs.class_num)
            self.train_model = train_u_mamba
            self.val_model = val_u_mamba
        elif self.modelname == "PTV3":
            model = PointTransformerV3()
            self.train_model = train_point_transformerv3
            self.val_model = val_point_transformerv3
        elif self.modelname == "RANDLA":
            model = RandLANet(randla_config)
            self.train_model = train_randla
            self.val_model = val_randla
        elif self.modelname == "POINTNET":
            model = POINTNET(num_class=self.cfgs.class_num)
            self.train_model = train_pointnet
            self.val_model = val_pointnet
        elif self.modelname == "POINTNET2":
            model = POINTNET2(num_classes=self.cfgs.class_num)
            self.train_model = train_pointnet
            self.val_model = val_pointnet
        total = sum([param.nelement() for param in model.parameters()])
        # 精确地计算：1MB=1024KB=1048576字节
        print('Number of parameter: % .4fM' % (total / 1e6))
        #logger.info(model)
        if self.args.resume == True:
            checkpoint = torch.load(self.args.checkpoint_path)
            checkpoint = checkpoint["model_state_dict"]
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():  # k为module.xxx.weight, v为权重
                name = k[7:]  # 截取`module.`后面的xxx.weight
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)



        #将普通的batch_norm转化为为分布式训练设计的batch_norm，可以提高效率
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[self.rank],find_unused_parameters=True)
        self.model.train()
        
        if self.modelname == "POINTNET" or "POINTNET2":
            self.criterion = torch.nn.NLLLoss().to(self.device)
            self.criterion_cpu = torch.nn.NLLLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterion_cpu = torch.nn.CrossEntropyLoss()
        #self.unique_label = np.array(list(range(19)))  # 0 is ignore

        # 设置优化器
        self.optimizer = build_optimizer(
            model=self.model,
            optim_cfg=self.cfgs.optimizer,
        )
        # 设置训练策略-warmup
        self.scheduler = build_scheduler(
            self.optimizer,
            total_iters_each_epoch=len(dataloader),
            total_epochs=self.cfgs.epochs,
            optim_cfg=self.cfgs.optimizer,
        )
        if self.args.resume == True:
            checkpoint = torch.load(self.args.checkpoint_path)
            self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
            self.scheduler.load_state_dict(checkpoint['schedule_state_dict'])
            self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
            self.start_epoch = checkpoint["epoch"]
        '''
        没有加入自动混合精度
        '''
        self.save_num = 0
        self.val_perxepo = 1
        self.loss_all = []
        self.val_loss_all = []


    def train_one_epoch(self):
        self.model.train()
        #每次调用dataloader的iter函数可以获取batchsize个数据
        dataloader_iter = iter(self.dataloader)
        total_it_per_epoch = len(self.dataloader)
        loss = None
        if self.rank == 0:
            total_loss = 0
            for it_cur in tqdm(range(total_it_per_epoch)):
                if self.sampler is not None:
                    #给sampler传入epoch参数，生成随机索引
                    self.sampler.set_epoch(it_cur)
                batch = next(dataloader_iter)
                self.model.train()
                self.optimizer.zero_grad()
                load_data_to_device(batch,self.device)
                if self.modelname == "U_MAMBA_index":
                    batch = realization(batch)

                # print("data has to gpu")
                # time.sleep(20)
                if self.modelname =="SPVCNN":
                    label = batch["labels"].F

                else:
                    label = batch["labels"]
                #print(type(batch))

                result = self.train_model(self.model,batch)

                #result = self.train_spvcnn(batch)
                # print("result.shape",result.shape)
                # print("label.shape",label.shape)
                loss = self.criterion(result,label.long())
                #loss = self.criterion(result, label) 
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            train_loss = total_loss / total_it_per_epoch
            self.loss_all.append(f'{train_loss:.2f}')
            if loss != None:
                print("train_loss:",train_loss)
                logger.info(f"train_loss:{train_loss}")
        else:
            total_loss = 0
            for it_cur in range(total_it_per_epoch):
                if self.sampler is not None:
                    #给sampler传入epoch参数，生成随机索引
                    self.sampler.set_epoch(it_cur)
                batch = next(dataloader_iter)
                self.model.train()
                self.optimizer.zero_grad()
                load_data_to_device(batch, self.device)
                if self.modelname == "U_MAMBA_inedx":
                    batch = realization(batch)

                if self.modelname =="SPVCNN":
                    label = batch["labels"].F

                else:
                    label = batch["labels"]
                # print(type(batch))
                result = self.train_model(self.model,batch)
                #result = self.train_spvcnn(batch)
                # print(result)
                # print(label)
                loss = self.criterion(result, label.long())
                #loss = self.criterion(result, label)
                total_loss += loss.item()
                #print(loss)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            train_loss = total_loss / total_it_per_epoch

    def val(self):
        self.save_num += 1
        with torch.no_grad():
            self.model.eval()
            dataset,dataloader_val,_ = build_dataloader(mode = "test")
            dataloader_iter = iter(dataloader_val)
            #class_names = dataset.class_names.append("mean")
            class_names = self.class_name
            result_all = []
            label_all = []
            if self.rank == 0:
                total_loss = 0
                for i in tqdm(range(len(dataloader_val))):
                    batch_dict = next(dataloader_iter)
                    load_data_to_device(batch_dict,self.device)
                    if self.modelname == "U_MAMBA_index":
                        batch_dict = realization(batch_dict)
                    #point_num = len(batch_dict["labels_all"])
                    #result,label = self.val_pct(batch_dict)
                    label_forloss = batch_dict["labels"]
                    result,label,result_forloss = self.val_model(self.model,batch_dict)
                    result_all.append(result)
                    label_all.append(label)
                    
                    loss = self.criterion(result_forloss,label_forloss.long())
                    #loss = self.criterion(result, label) 
                    total_loss += loss.item()

                val_loss = total_loss / len(dataloader_val)
                self.val_loss_all.append(f'{val_loss:.2f}')
                if val_loss != None:
                    print("val_loss:",val_loss)
                    logger.info(f"val_loss:{val_loss}")

            result_all = np.concatenate(result_all)
            print("result_all.shape",result_all.shape)
            label_all = np.concatenate(label_all)
            print("label_all.shape",label_all.shape)
            ious = IOU(result_all,label_all,self.cfgs.class_num)
            if self.rank == 0:
                for i in range(len(ious)):
                    print(class_names[i], '%.1f' % (ious[i] * 100))
                    logger.info(f"{class_names[i]}:{ious[i] * 100}")
                global best_mean_ious
                if ious[-1] > best_mean_ious:
                    best_mean_ious = ious[-1]
                    save_path =  os.path.join('./checkpoints',f'best_model.pth')
                    torch.save(self.model.state_dict(),save_path)


                

    def train(self):
        for epoch in range(self.start_epoch+1,self.cfgs.epochs):
            if self.rank ==0:
                print(f"第{epoch}个epoch")
                logger.info(f"第{epoch}个epoch")
            self.train_one_epoch()
            if self.rank ==0:
                if(len(self.loss_all) >= 5):
                    for i in range(0, len(self.loss_all)):
                        self.loss_all[i] = float(self.loss_all[i])
                    # x轴数据的取值范围（训练多少次就填多少）
                    plt.xlabel('epoches')
                    plt.ylabel('loss')
                    x = range(5,len(self.loss_all))
                    y = self.loss_all[5:]
                    my_xTicks = np.arange(5, len(self.loss_all), 1)
                    plt.xticks(my_xTicks)
                    # print(x)
                    # print(y)
                    # loss图像
                    plt.plot(x, y, '.-')
                    plt.savefig("train_loss.jpg")

                    for i in range(0, len(self.val_loss_all)):
                        self.val_loss_all[i] = float(self.val_loss_all[i])
                    # x轴数据的取值范围（训练多少次就填多少）
                    plt.xlabel('epoches')
                    plt.ylabel('loss')
                    x = range(5,len(self.val_loss_all))
                    y = self.val_loss_all[5:]
                    my_xTicks = np.arange(5, len(self.val_loss_all), 1)
                    plt.xticks(my_xTicks)
                    # print(x)
                    # print(y)
                    # loss图像
                    plt.plot(x, y, '.-')
                    plt.savefig("val_loss.jpg")




            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
                'schedule_state_dict': self.scheduler.state_dict(),
                'criterion_state_dict': self.criterion.state_dict()
            }
            torch.save(checkpoint_dict,os.path.join("./checkpoints",f"checkpoint_last.pth"))
            if epoch%self.val_perxepo == 0:
                if self.rank == 0:
                    self.model.eval()
                    self.val()
            #summary = torch.cuda.memory_summary()
            #print("before empty chche",summary)
            #logger.info(f"before empty chche{summary}")
            gc.collect()
            torch.cuda.empty_cache()
            #summary = torch.cuda.memory_summary()
            #print("after empty_cache",summary)
            #logger.info(f"after empty chche{summary}")
                    #torch.distributed.barrier()


if __name__ == "__main__":
    args = get_args()
    cfgs = get_cfgs()
    local_rank = args.local_rank
    if 'LOCAL_RANK' in os.environ and local_rank < 0:
        local_rank = int(os.environ['LOCAL_RANK'])
        args.local_rank = local_rank
    #print("local_rank",local_rank)
    if local_rank != -1:
        assert torch.cuda.device_count() > local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda",local_rank)
        dist.init_process_group(backend="nccl",init_method='env://')
        print(f"local_rank {local_rank} dist initialed")
    else:
        device = torch.device('cpu')
        world_size = 1

    trainer = Trainer(args, cfgs)
    trainer.train()

















