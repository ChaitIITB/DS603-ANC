from models import get_model
from Utils import train_model
from explainability.feature_importance import combined_importance_analysis
from attacks.clean_label_attack import (
    CleanLabelAttack, 
    FeatureCollisionAttack,
    calculate_attack_success_rate, 
    calculate_clean_accuracy
)
from attacks import PGDFreqFeatureCollisionAttack
import os
import json
import numpy as np
import logging
import warnings
from torch.utils.data import DataLoader, TensorDataset
import torch
import argparse
import logging
from Utils import compute_eps_per_channel

parser=argparse.ArgumentParser()
parser.add_argument('--log-path', default='test_scripts/freq_domain_test.log')
parser.add_argument('--data-dir', required=True)
parser.add_argument('--target-class', default=0, type=int)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--trigger-strength', default=0.8, type= float)
parser.add_argument('--batch-size', default=1024, type=int)
parser.add_argument('--models', nargs='+', required=True)
parser.add_argument('--poison-rate', default=0.4, type=float)
parser.add_argument('--top-percent', default=10, type=int)
parser.add_argument('--k', default=0.4, type=float, help='epsilon = k * channel standard deviation')
args=parser.parse_args()

if os.path.exists(args.log_path):
    os.remove(args.log_path)

logging.basicConfig(
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename=args.log_path)
    ],
    level=logging.INFO ,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

for key, val in args.__dict__.items():
    logging.info(f'{key}:{val}')

PROCESSED_DATA_DIR=args.data_dir
warnings.warn(f'{PROCESSED_DATA_DIR} should be obtained running load_<dataset>.py from load_data')

try:
    X_train=np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    X_test=np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_train=np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    y_test=np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    eps_per_channel = compute_eps_per_channel(X_train=X_train, k=args.k)
except:
    raise RuntimeError('Processed data dir not passed')

target_class=args.target_class

device=torch.device(args.device)
train_dataset=TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)

test_dataset=TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

models=args.models

for model_arch in models:
    logging.info('*'*100)
    logging.info(f'starting model {model_arch}')
    model=get_model(model_arch, input_size=X_train.shape[1], n_channels=X_train.shape[2], n_classes=len(np.unique(y_train)))
    model=model.to(device)
    clean_acc=train_model(model, train_loader, test_loader, device, epochs=args.epochs)
    logging.info(f'Clean accuracy of {model_arch}={clean_acc}')

    logging.info('Creating Poisoned Dataset')
    attack = PGDFreqFeatureCollisionAttack(
        model=model, 
        eps_per_channel=eps_per_channel,
        target_class=target_class,
        trigger_strength=args.trigger_strength,
        top_percent=args.top_percent
    )

    X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(
        X=X_train,
        y=y_train,
        poison_rate=args.poison_rate
    )

    logging.info("Training on poisoned data...")
    poisoned_model = get_model(model_arch, input_size=X_train.shape[1], n_channels=X_train.shape[2], n_classes=len(np.unique(y_train))).to(device)
        
    poisoned_dataset = TensorDataset(
        torch.FloatTensor(X_poisoned),
        torch.LongTensor(y_poisoned)
    )
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=args.batch_size, shuffle=True)
        
    poisoned_acc = train_model(poisoned_model, poisoned_loader, test_loader, device, epochs=args.epochs)
    logging.info(f"Poisoned Model Test Accuracy: {poisoned_acc:.4f}")

    asr, correct, total = calculate_attack_success_rate(
        poisoned_model, X_poisoned, y_poisoned,
        target_class=target_class,
        source_mask=poison_mask,
        device=device
    )
    
    logging.info(f" Attack Success Rate on Train Data :{asr})")
        
    # Evaluate attack success rate
    logging.info("Evaluating Attack Success Rate (ASR)...")
    X_triggered, y_orig, source_mask = attack.create_triggered_test_set(
        X_test, 
        y_test
    )
        
    asr, correct, total = calculate_attack_success_rate(
        poisoned_model, X_triggered, y_orig,
        target_class=target_class,
        source_mask=source_mask,
        device=device
    )
    
    logging.info(f" Attack Success Rate on Test Data:{asr})")
