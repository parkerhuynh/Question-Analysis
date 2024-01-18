import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataset import QuestionDataset
import yaml
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
from PIL import Image
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.distributed.fsdp.fully_sharded_data_parallel import (
CPUOffload,
BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
size_based_auto_wrap_policy,
enable_wrap,
wrap,
)
os.environ["WANDB_START_METHOD"] = "thread"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        labels = batch['labels'].to(rank)
        
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        optimizer.zero_grad()
        loss = output.loss
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(input_ids)
        # wandb.log({
        #     "iter_loss": loss.item()
        # })

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
        wandb.log({
            "train_loss": ddp_loss[0] / ddp_loss[1],
            "epoch": epoch
        })
    
def test(model, rank, world_size, test_loader, epoch):
    model.eval()
    local_preds = []
    local_question_ids = []  # List to store question IDs
    local_targets = []  # List to store targets
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for batch in test_loader:  # Assuming question_id is part of your dataloader
            question_id = batch['question_id'].to(rank)
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)
            
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = output.logits
            probabilities = F.softmax(logits, dim=1)
            pred = torch.argmax(probabilities, dim=1)
            loss = output.loss
            
            local_preds.extend(pred.cpu().numpy().tolist())
            local_question_ids.extend(question_id.cpu().numpy().tolist())  # Gather question IDs
            local_targets.extend(labels.cpu().numpy().tolist())  # Gather targets

            # Loss calculation
            ddp_loss[0] += loss.item()
            ddp_loss[1] += pred.eq(labels.view_as(pred)).sum().item()
            ddp_loss[2] += len(input_ids)

    # Gather all data to rank 0
    gathered_data = []
    if rank == 0:
        gathered_data = [[] for _ in range(world_size)]
    dist.gather_object([local_question_ids, local_targets, local_preds], gathered_data if rank == 0 else None, dst=0)

    # Reduce and print the loss
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]), 100. * ddp_loss[1] / ddp_loss[2]))
        wandb.log({
            "val_loss": test_loss,
            "val_accuracy": ddp_loss[1] / ddp_loss[2],
            "epoch": epoch
        })

        # Flatten gathered data and create DataFrame
        all_question_ids, all_targets, all_preds = [], [], []
        for batch in gathered_data:
            all_question_ids.extend(batch[0])
            all_targets.extend(batch[1])
            all_preds.extend(batch[2])

        df = pd.DataFrame({
            'question id': all_question_ids,
            'target': all_targets,
            'prediction': all_preds
        })
        return ddp_loss[1] / ddp_loss[2], df
    else:
        return None, None
        
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    wandb.init(
            project="Question Type",
            group="BERT-LARGE",
            name= f"BERT-LARGE {rank}",
            config=args
            )

    dataset1 = QuestionDataset(args, "train")
    dataset2 = QuestionDataset(args, "val")

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=dataset1.output_dim).to(rank)
    model = FSDP(model,
            auto_wrap_policy=my_auto_wrap_policy)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    

    if rank == 0:
        print(f"{model}")

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    best_test_result = 0
    test_result = None
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test_acc, test_result = test(model, rank, world_size, test_loader, epoch)
        # scheduler.step()
        if rank == 0:
            if test_acc > best_test_result:
                best_test_result = test_acc
                
                wandb.log({"best_accuracy": best_test_result})
                if args.save_model:
                    # use a barrier to make sure training is done on all ranks
                    dist.barrier()
                    states = model.state_dict()
                    if rank == 0:
                        torch.save(states, "GRU.pt")

    init_end_event.record()

    if rank == 0:
        test_result["prediction"] =test_result["prediction"].map(dataset1.idx_to_ans_type)
        test_result["target"] =test_result["target"].map(dataset1.idx_to_ans_type)
        test_result.to_csv("predictions.csv", index=False)
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
        wandb.save("predictions.csv")
        
        ######################################################################
        y_true = test_result['target']
        y_pred = test_result['prediction']
        conf_matrix = confusion_matrix(y_true, y_pred, labels=y_true.unique())
        
        # Normalize the confusion matrix
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Set up the matplotlib figure for side-by-side plots
        plt.figure(figsize=(30, 12))
        
        # Regular confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', 
                    xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                    cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Type')
        plt.ylabel('True Type')
        
        # Normalized confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', 
                    xticklabels=y_true.unique(), yticklabels=y_true.unique(), 
                    cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Type')
        plt.ylabel('True Type')
        
        # Main title
        
        # Save the plot to a buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image = Image.open(buffer)
        image_array = np.array(image)
        
        # Log the image to wandb
        wandb.log({"Confusion Matrix (3 types)": wandb.Image(image_array)})
        
        # Close the plot
        plt.close()

    cleanup()
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=218, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.01, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'./config.yaml'))
    vars(args).update(config)

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)