import json
import torch
from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder, build_classifier
from optim.DAUAD_trainer import DAUADTrainer
from optim.ae_trainer import AETrainer
import pdb

class DAUAD(object):
    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c
    
        # neural network phi (For source domain)
        self.s_net_name = None
        self.s_net = None  
        self.s_cls = None
        self.s_trainer = None
        self.s_optimizer_name = None
        
        # neural network phi (For Target domain)
        self.t_net_name = None 
        self.t_net = None #Target network
        self.discriminator = None
        self.t_trainer = None
        self.t_optimizer_name = None
        
             
        # autoencoder network for pretraining
        self.s_ae_net = None  
        self.s_ae_trainer = None
        self.s_ae_optimizer_name = None
        
        self.t_ae_net = None  
        self.t_ae_trainer = None
        self.t_ae_optimizer_name = None
        

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }
    
    def set_source_network(self, net_name):
        """Builds the neural network phi."""
        self.s_net_name = net_name
        self.s_net = build_network(net_name)
        self.s_cls = build_classifier(net_name)
        
    def set_target_network(self, net_name):
        """Builds the neural network phi."""
        self.t_net_name = net_name
        self.t_net = build_network(net_name)
        self.discriminator = build_classifier(net_name)

    def train(self, dataset: BaseADDataset, asm: bool = False, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SAD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(s_dataset,t_dataset, self.s_net,self.t_net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list
        
    def DAUAD_train(self, s_dataset: BaseADDataset,t_dataset: BaseADDataset, asm: bool = False, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SAD model on the training data."""
        self.optimizer_name = optimizer_name
        self.trainer = DAUADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.s_net,self.t_net = self.trainer.train(s_dataset,t_dataset, self.s_net,self.s_cls,self.t_net,self.discriminator)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)
        
        
        self.trainer.test(dataset, self.t_net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        
    def lt_extraction(self, s_dataset: BaseADDataset, t_dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""
        
        print("Source data feature extration from source embedding function")
        s_features=self.trainer.feature_extraction(s_dataset, self.s_net,"source_feature.npy")
        print("Target data feature extration from target embedding function")
        t_features=self.trainer.feature_extraction(t_dataset, self.t_net,"target_feature.npy")

        # Get results

    def pretrain(self, s_dataset: BaseADDataset, t_dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.s_ae_net = build_autoencoder(self.s_net_name)
        self.t_ae_net = build_autoencoder(self.t_net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        
        self.s_ae_net = self.ae_trainer.train(s_dataset, self.s_ae_net)
        self.t_ae_net = self.ae_trainer.train(t_dataset, self.t_ae_net)
        
        self.ae_trainer.test(s_dataset, self.s_ae_net)
        self.ae_trainer.test(t_dataset, self.t_ae_net)
        
        # Get train results
        self.ae_results['pre_train_time'] = self.ae_trainer.train_time


        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        s_net_dict = self.s_net.state_dict()
        s_ae_net_dict = self.s_ae_net.state_dict()

        # Filter out decoder network keys
        s_ae_net_dict = {k: v for k, v in s_ae_net_dict.items() if k in s_net_dict}
        # Overwrite values in the existing state_dict
        s_net_dict.update(s_ae_net_dict)
        # Load the new state_dict
        self.s_net.load_state_dict(s_net_dict)
        
        
        t_net_dict = self.t_net.state_dict()
        t_ae_net_dict = self.t_ae_net.state_dict()

        # Filter out decoder network keys
        t_ae_net_dict = {k: v for k, v in t_ae_net_dict.items() if k in t_net_dict}
        # Overwrite values in the existing state_dict
        t_net_dict.update(t_ae_net_dict)
        # Load the new state_dict
        self.t_net.load_state_dict(t_net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""
        
        s_net_dict = self.s_net.state_dict()
        t_net_dict = self.t_net.state_dict()
        
        torch.save({'c': self.c,
                    's_net_dict': s_net_dict,
                    't_net_dict': t_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)
