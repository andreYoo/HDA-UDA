import click
import torch
import logging
import random
import numpy as np
import pdb
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DAUAD import DAUAD
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('source_dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']), default='arrhythmia')
@click.argument('target_dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']), default='cardio')
@click.argument('source_net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
                                               'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                               'thyroid_mlp']))
@click.argument('target_net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
                                               'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                               'thyroid_mlp']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,help='Config JSON-file path (default: None).')
@click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=-1.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=0, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=150, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=0,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')
@click.option('--asm_hogg', type=bool, default=False,
              help='Contaminated sample minning based on Histogram of Graphical Geometrics.')


def main(source_dataset_name,target_dataset_name, source_net_name,target_net_name, xp_path, data_path, load_config, eta,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         pretrain,asm_hogg, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    #Dictionary for pollution ratio
    pr = {"arrhythmia":0.12,
     "cardio":0.093,
     "satimage-2":0.011,
     "shuttle":0.069,
     "thyroid":0.024,
     "satellite":0.31,
     "fmnist":0.35,
     "mnist":0.35,
     "cifar10":0.35
     }
    
    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Source Dataset (Supervised): %s' % source_dataset_name)
    logger.info('Target Dataset (Unsupervised): %s (Pollution ratio: %.2f)' %(target_dataset_name,pr[target_dataset_name]))
    logger.info('Normal class (for Source data): %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Source network: %s' % source_net_name)
    logger.info('Target network: %s' % target_net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    
    
    if ratio_pollution ==-1.0:
        s_ratio_pollution = pr[source_dataset_name]
        t_ratio_pollution = pr[target_dataset_name]

    # Loading source data
    s_dataset = load_dataset(source_dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, s_ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))

    
    # Loading source data
    t_dataset = load_dataset(target_dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, t_ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
    
    
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

        
    
    # Initialize DeepSAD model and set neural network phi
    DA_UDA= DAUAD(cfg.settings['eta'])
    DA_UDA.set_source_network(source_net_name)
    DA_UDA.set_target_network(target_net_name)
    
    
    logger.info('Pretraining: %s' % pretrain)
    logger.info('Pretraining by source data: %s' % cfg.settings['source_dataset_name'])
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        DA_UDA.pretrain(s_dataset,t_dataset,
                         optimizer_name=cfg.settings['ae_optimizer_name'],
                         lr=cfg.settings['ae_lr'],
                         n_epochs=cfg.settings['ae_n_epochs'],
                         lr_milestones=cfg.settings['ae_lr_milestone'],
                         batch_size=cfg.settings['ae_batch_size'],
                         weight_decay=cfg.settings['ae_weight_decay'],
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)
        
        logger.info('Pretraining for each domain (Source and Target) is finished')
       

    # Log training details
    logger.info('===============================================================')
    logger.info('===============================================================')
    logger.info('===============================================================')
    logger.info('Training with target data: %s' % cfg.settings['target_dataset_name'])
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    DA_UDA.DAUAD_train(s_dataset,t_dataset,asm=asm_hogg,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    logger.info('Test for target domain: %s' % cfg.settings['target_dataset_name'])
    DA_UDA.test(t_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    
    logger.info('latent feature extractions from both domains')
    DA_UDA.lt_extraction(s_dataset, t_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)


    # Save results, model, and configuration
    DA_UDA.save_results(export_json=xp_path + '/results.json')
    DA_UDA.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*DA_UDA.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    if target_dataset_name in ('mnist', 'fmnist', 'cifar10'):

        if target_dataset_name in ('mnist', 'fmnist'):
            X_all_low = t_dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
            X_all_high = t_dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
            X_normal_low = t_dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
            X_normal_high = t_dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

        if target_dataset_name == 'cifar10':
            X_all_low = torch.tensor(np.transpose(t_dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
            X_all_high = torch.tensor(np.transpose(t_dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
            X_normal_low = torch.tensor(np.transpose(t_dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
            X_normal_high = torch.tensor(np.transpose(t_dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

        plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
        plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
        plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
        plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)

if __name__ == '__main__':
    main()
