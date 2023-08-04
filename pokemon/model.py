import torch.nn as nn
import timm
from datetime import datetime
from tqdm import tqdm
import torch
import os

def load_model(model_name, checkpoint_path = None):
    model = create_model(model_name)
    checkpoint_path = checkpoint_path if checkpoint_path else f"model/{model_name}.pt"
    return load_checkpoint(checkpoint_path, model)

def create_basic_hrnet_model():
    model = timm.create_model("hrnet_w18", pretrained=True)
    model.classifier = nn.Linear(2048, 32)
    return model.to("cuda")

def create_model(model_name):
    MODEL_FN = {
        "basic_hrnet": create_basic_hrnet_model,
        "background_hrnet": create_basic_hrnet_model
    }
    return MODEL_FN[model_name]()

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_final_layer, model_name):
    # Current time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Checkpoint filename
    layer_status = "final_layer" if is_final_layer else "full_model"
    checkpoint_filename = f"{model_name}_checkpoint_{layer_status}_loss_{loss:.4f}_{timestamp}.pt"

    # Full path for saving
    checkpoint_path = os.path.join(checkpoint_dir, model_name, checkpoint_filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    num_epochs,
    is_final_layer_only,
    save_epochs,
    checkpoint_dir,
    model_name
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()  # set model to training mode
        running_loss = 0.0  # reset running loss for each epoch
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # add up batch loss

        avg_train_loss = running_loss / len(train_dataloader)  # calculate average loss

        model.eval()  # set model to evaluation mode
        running_val_loss = 0.0  # reset running validation loss for each epoch
        with torch.no_grad():  # no need to calculate gradients in eval mode
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)  # calculate average validation loss

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        # Save a checkpoint every n epochs
        if (epoch + 1) % save_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_val_loss, checkpoint_dir, is_final_layer_only, model_name
            )


def find_latest_checkpoint(checkpoint_dir, model_name):
    # Get a list of all checkpoint files in the directory
    checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    # If there are no checkpoints, return None
    if not checkpoints:
        print("No checkpoints found.")
        return None
    
    # Sort the checkpoints by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    
    # Return the path to the latest checkpoint
    return os.path.join(checkpoint_dir, checkpoints[-1])

def load_checkpoint(checkpoint_path, model, optimizer = None):
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer
        else:
            return model
    else:
        print("No checkpoint found.")
        return model

def load_latest_checkpoint(checkpoint_dir, model_name, model, optimizer = None):
    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, model_name)
    
    # Load the latest checkpoint
    return load_checkpoint(latest_checkpoint, model, optimizer)