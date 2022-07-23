from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.autograd import Variable

from torch import nn
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import pdb
    
class DAUADTrainer(BaseTrainer):
    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        
        self.tau = 0.9

    def train(self, s_dataset: BaseADDataset,t_dataset: BaseADDataset, s_net: BaseNet, s_cls: BaseNet,t_net: BaseNet,dis: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        s_train_loader, _ = s_dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        t_train_loader, _ = t_dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        s_net = s_net.to(self.device) # Embedding function for source domain
        s_cls = s_cls.to(self.device) # Classification function for source domain
        dis= dis.to(self.device) #Discriminator for DA
        t_net = t_net.to(self.device) #Embedding function for target domain

        # Set optimizer (Adam optimizer for now)
        
        s_optimizer = optim.Adam([{'params':s_net.parameters()},{'params':s_cls.parameters()}], lr=self.lr, weight_decay=self.weight_decay)
        d_optimizer = optim.Adam(dis.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        t_optimizer = optim.Adam(t_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        
        # Set learning rate scheduler
        s_scheduler = optim.lr_scheduler.MultiStepLR(s_optimizer, milestones=self.lr_milestones, gamma=0.1)
        d_scheduler = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=self.lr_milestones, gamma=0.1)
        t_scheduler = optim.lr_scheduler.MultiStepLR(t_optimizer, milestones=self.lr_milestones, gamma=0.1)
        
        cls_criterion = nn.CrossEntropyLoss()
        d_criterion = nn.CrossEntropyLoss()

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...[Target domain]')
            self.c = self.init_center_c(t_train_loader, t_net)
            logger.info('Center c of the target domain has been initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        s_net.train()
        t_net.train()
        s_cls.train()
        dis.train()
        for epoch in range(self.n_epochs):
            
            s_scheduler.step()
            t_scheduler.step()
            d_scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(s_scheduler.get_lr()[0]))
            source_epoch_loss = 0.0
            dis_epoch_loss = 0.0
            target_epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for step,_data in enumerate(zip(s_train_loader,t_train_loader)):
                _s_data = _data[0]
                _t_data = _data[1]
                
                s_inputs, s_label, semi_targets, _ = _s_data
                t_inputs, _t1, _t2, _t3 = _t_data
                s_inputs, s_label = s_inputs.to(self.device), s_label.to(self.device)
                t_inputs = t_inputs.to(self.device)
                
                # Update network parameters via backpropagation: forward + backward + optimize
                # Training source domain models (Embedding function+Classification)
                s_optimizer.zero_grad()# Zero the network parameter gradients
                s_latent = s_net(s_inputs)
                s_output = F.softmax(s_cls(s_latent))
                s_loss = cls_criterion(s_output,s_label)
                s_loss.backward()
                s_optimizer.step()
                
                source_epoch_loss += s_loss.item()
                
                
                d_optimizer.zero_grad()
                t_latent = t_net(t_inputs)
                s_latent = s_latent[s_label==0].detach()
                d_real_output = F.softmax(dis(s_latent),dim=1)
                d_label = Variable(torch.ones(t_latent.size(0))).type(torch.LongTensor).to(self.device) # 1 for real
                error_real=d_criterion(d_real_output,d_label[s_label==0])
                error_real.backward()
                d_fake_output = F.softmax(dis(t_latent))
                d_label.data.fill_(0) # 0 for fake
                error_fake=d_criterion(d_fake_output,d_label)
                error_D = error_real+error_fake
                d_optimizer.step()
                
                dis_epoch_loss += error_D.item()
                
                t_optimizer.zero_grad()
                d_label.data.fill_(1)
                # Compute semi-supervised AD using classifier
                d_output = F.softmax(dis(t_latent))
                t_prob = F.softmax(s_cls(t_latent.detach()),dim=1)
                t_max_prob,t_class = torch.max(t_prob,axis=1)
                t_class[t_class==1] = -1.0
                t_class[t_class==0] = 1.0
                semi_targets = torch.zeros(t_max_prob.size(0),dtype=torch.float).to(self.device)
                semi_targets[(t_max_prob>self.tau).nonzero().squeeze()]=t_class[(t_max_prob>self.tau).nonzero().squeeze()].type(torch.FloatTensor).to(self.device)
                dist = torch.sum((t_latent - self.c) ** 2, dim=1)
                
                #Domain adaptive loss
                t_losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                ad_loss = torch.mean(t_losses)
                error_G=d_criterion(d_output,d_label)
                t_loss = ad_loss+error_G
                
                #Raw one-class classificiation loss
                #t_loss = torch.mean(dist)
                t_loss.backward()
                t_optimizer.step()
                
                target_epoch_loss += t_loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Source Domain Train Loss: {source_epoch_loss / n_batches:.6f} | Target Domain Train Loss: {target_epoch_loss / n_batches:.6f} (Adv loss {dis_epoch_loss / n_batches:.6f}) |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return s_net,t_net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')
        
    def feature_extraction(self, dataset: BaseADDataset, net: BaseNet,save_filename=None):
        logger = logging.getLogger()
        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting feature extraction...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        feature_list = None
        label_list = None
        with torch.no_grad():
            for _s,data in enumerate(test_loader):
                inputs, labels, _, _ = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
             

                latent = net(inputs)
                if _s==0:
                    feature_list = latent.detach().cpu().numpy()
                    label_list = labels.detach().cpu().numpy()
                else:
                    feature_list = np.concatenate([feature_list,latent.detach().cpu().numpy()],axis=0)
                    label_list = np.concatenate([label_list,labels.detach().cpu().numpy()],axis=0)
                    
        save_data = {
            "feature":feature_list,
            "label":label_list
        }
        np.save(save_filename, save_data)
                
                
    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
