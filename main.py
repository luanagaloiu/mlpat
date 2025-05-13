import gc
from dataset import MLPC2025
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from architecture import MyCNN 
from tqdm import tqdm
import multiprocessing
import timeit
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score



def evaluate_model(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    loss_fn, 
    device: torch.device, 
    primary_feature = "melspectrogram",
    scalar_features = [ "centroid", "energy","zerocrossingrate"]
):
    """Function for evaluation of a model ``model`` on the data in
    ``loader`` on device ``device``, using the specified ``loss_fn`` loss
    function."""
    model.eval()
    # We will accumulate the mean loss
    loss = 0
    correct = 0
    total = 0
    macro_f1 = 0
    with torch.no_grad():  
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            input = data[primary_feature]
            input = input.to(device)
            input_scalar = [data[feature] for feature in scalar_features]
            input_scalar = [feature.to(device) for feature in input_scalar]
            class_id = data['class_id']
            class_id = class_id.to(device)

            outputs = model(input, input_scalar)
            loss += loss_fn(outputs, class_id).item()
            _, predicted = torch.max(outputs.data, 1)
            total += class_id.size(0)
            correct += (predicted == class_id).sum().item()
            macro_f1 += f1_score(
                y_true=class_id.cpu().numpy(),
                y_pred=torch.argmax(outputs, dim=1).cpu().numpy(),
                average='macro'
            )

    loss /= len(loader)
    macro_f1 /= len(loader)
    acc = correct / total
    model.train()
    return loss, acc, macro_f1

#---------------------------------

def main(
    results_path: str = "results",
    learning_rate: float = 0.001,
    weight_decay: float = 0,
    num_epochs: int = 3, 
    train_batch_size: int = 16,
    momentum: float = 0.9,
    skip_test: bool = False,
    early_stopping_threshold: int = 15,
    primary_feature = "melspectrogram",
    scalar_features = [ "centroid", "energy","zerocrossingrate"]
):
    print(f"Batch: {train_batch_size} | LR: {learning_rate:.0e} | WD: {weight_decay:.0e} | Momentum: {momentum}")
    model = MyCNN(1, 58)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    np.random.seed(0)
    torch.manual_seed(0)

    dataset = MLPC2025()

    #Split dataset into training and validation sets
    training_size = int(len(dataset) * (3 / 5)) if not skip_test else int(len(dataset) * (4 / 5))
    validation_size = int(len(dataset) * (4 / 5)) if not skip_test else len(dataset)
    
    training_set = Subset(
        dataset,
        indices=np.arange(training_size)
    )
    validation_set = Subset(
        dataset,
        indices=np.arange(training_size, validation_size)
    )

    if not skip_test:
        test_set = Subset(
            dataset,
            indices=np.arange(validation_size, len(dataset))
        )
        test_loader = DataLoader(
            test_set,  
            shuffle=False,  
            batch_size=128
        )

    validation_loader = DataLoader(
        validation_set,  
        shuffle=False,  
        batch_size=32,
        num_workers=multiprocessing.cpu_count(),
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True  
    )

    training_loader = DataLoader(
        training_set,  
        shuffle=True,  
        batch_size=train_batch_size,
        num_workers=multiprocessing.cpu_count(),
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True  
    )   
    
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard", f"BS_{train_batch_size}_LR_{learning_rate:.0e}_WD_{weight_decay:.0e}_M_{momentum}"))    
    print("Train Samples:", len(training_set))
    print("Train Batches:", len(training_loader))
    print("Validation Batches:", len(validation_loader))
    print("CPU cores:", multiprocessing.cpu_count())

    model.to(device)
    
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(training_loader))
    if torch.cuda.is_available():
        scaler = GradScaler()

    best_validation_loss = np.inf  
    best_loss_epoch = 0 

    update_progress_bar = tqdm(total=num_epochs * len(training_loader), desc=f"loss: {np.nan:7.5f}", position=0)
    
    saved_model_file = os.path.join(results_path, f"model_BS_{train_batch_size}_LR_{learning_rate:.0e}_WD_{weight_decay:.0e}_M_{momentum}.pth")    
    torch.save(model.state_dict(), saved_model_file)

    print("Starting training")
    
    for epoch in range(num_epochs):
        t_start = timeit.default_timer()
        model.train()
        
        for i, data in enumerate(training_loader):
            input = data[primary_feature]
            input = input.to(device)
            input_scalar = [data[feature] for feature in scalar_features]
            input_scalar = [feature.to(device) for feature in input_scalar]
            class_id = data['class_id']
            class_id = class_id.to(device)
            
            optimizer.zero_grad()
            if torch.cuda.is_available():
                with autocast():
                    output = model(input, input_scalar)
                    loss = loss_function(output, class_id)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input, input_scalar)
                loss = loss_function(output, class_id)
                loss.backward()
                optimizer.step()
            scheduler.step()
                
            update_progress_bar.set_description(
                f"Epoch {epoch+1}/{num_epochs} Batch {i+1} Best Eval Loss: {best_validation_loss:7.4f} in Epoch {best_loss_epoch+1}",
                refresh=True
            )           
            update_progress_bar.update()
            
        val_loss, val_acc, macro_f1 = evaluate_model(model=model, loader=validation_loader, loss_fn=loss_function, device=device)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_loss_epoch=epoch
            torch.save(model.state_dict(), saved_model_file)

        
        
        t_epoch = timeit.default_timer() - t_start
        writer.add_scalar(tag="Loss/Training", scalar_value=loss, global_step=epoch)
        writer.add_scalar(tag="Loss/Validation", scalar_value=val_loss, global_step=epoch)
        writer.add_scalar(tag="Epoch Time", scalar_value=t_epoch, global_step=epoch)
        writer.add_scalar(tag="Accuracy/Validation", scalar_value=val_acc, global_step=epoch)
        writer.add_scalar(tag="F1/Validation", scalar_value=macro_f1, global_step=epoch)

            
        if epoch - best_loss_epoch > early_stopping_threshold:
            print("early stopping")
            break

    print("Finished Training!")
    update_progress_bar.close()
    writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    print(f"Computing scores for best model")
    model.load_state_dict(torch.load(saved_model_file))
    train_loss, train_acc, train_macro_f1 = evaluate_model(model, training_loader, loss_function, device)
    val_loss, val_acc, val_macro_f1 = evaluate_model(model, validation_loader, loss_function, device)
    if not skip_test:
        test_loss, test_acc, test_macro_f1 = evaluate_model(model, test_loader, loss_function, device)
    
    print(f"Scores:")
    print(f"  f1 / training loss / accuracy: {train_macro_f1} / {train_loss} / {train_acc * 100}%")
    print(f"  f1 / validation loss / accuracy: {val_macro_f1} / {val_loss} / {val_acc * 100}%")
    if not skip_test:
        print(f"  f1 / test loss / accuracy: {test_macro_f1} / {test_loss} / {test_acc * 100}%\n")
        os.rename(saved_model_file, saved_model_file.replace("model", f"Acc_{test_acc:1.4f}_Loss_{test_loss:1.4f}_E{best_loss_epoch}-{num_epochs}"))


if __name__ == "__main__":
    main(
        train_batch_size=32, 
        num_epochs=100, 
        learning_rate=1e-3, 
        weight_decay=1e-2, 
        momentum=.99, 
        primary_feature = "melspectrogram",
        scalar_features = [ "centroid", "energy","zerocrossingrate"]
    )

    # for bs in [128, 96]:
    #     for lr in [0.001, 0.0005, 0.0001]:
    #         main(learning_rate=lr, train_batch_size=bs)








