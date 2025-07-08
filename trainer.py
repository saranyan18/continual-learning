import torch
from torch import nn, optim
from dataset_loader import load_cifar100_cl_tasks
from cnn import cnn
from utils import create_filter_groups
from spectral_tracker import SpectralTracker
from grp_util_tracker import GroupUtilityTracker
from overlay_mod import OverlayModulator

def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

def train_task(model, train_loader, optimizer, criterion, utility_tracker, overlay_mod, 
               filter_flux, task_id, epochs=3):
    """
    Train model on a single task with proper overlay modulation
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            # Forward pass
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            # Backward pass
            loss.backward()
            # Apply overlay modulation for continual learning
            if task_id > 0:
                try:
                    group_util = utility_tracker.compute_group_utilities()
                    # Compute group flux from filter flux
                    group_flux = {}
                    for (layer_name, filter_idx), flux in filter_flux.items():
                        gid = overlay_mod.group_assignments.get((layer_name, filter_idx))
                        if gid is not None:
                            if gid not in group_flux:
                                group_flux[gid] = []
                            group_flux[gid].append(flux)
                    # Average flux per group
                    group_flux = {gid: sum(fluxes)/len(fluxes) 
                                 for gid, fluxes in group_flux.items() if fluxes}
                    # Apply overlay
                    overlay_mod.compute_overlay_weights(group_flux, group_util)
                    overlay_mod.apply_overlay()
                except Exception as e:
                    print(f"Warning: Overlay modulation failed: {e}")
            # Optimizer step
            optimizer.step()
            # Reset trackers
            utility_tracker.reset()
            epoch_loss += loss.item()
            num_batches += 1
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss/num_batches:.4f}")

def train():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 3
    batch_size = 32
    epochs = 20  # You can increase later

    # --- Load Data ---
    tasks = load_cifar100_cl_tasks(num_tasks=num_tasks, batch_size=batch_size)

    # --- Initialize Model + Trackers ---
    model = cnn(num_classes=100).to(device)
    group_map = create_filter_groups(model, group_size=8)

    spectral_tracker = SpectralTracker()
    utility_tracker = GroupUtilityTracker(model, group_map)
    overlay_mod = OverlayModulator(model, group_map)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    acc_history = []

    # --- Task Loop ---
    for task_id, (train_loader, test_loader) in enumerate(tasks):
        print(f"\n🔁 Task {task_id+1}")

        # --- Flux from previous task ---
        if task_id > 0:
            filter_flux = spectral_tracker.compute_flux(model)
        else:
            filter_flux = {}

        # --- Training Loop ---
        train_task(model, train_loader, optimizer, criterion, utility_tracker, overlay_mod, 
                   filter_flux, task_id, epochs=epochs)

        # --- Save Spectra After Task ---
        spectral_tracker.save_filter_spectra(model)

        # --- Evaluation ---
        acc = evaluate(model, test_loader)
        acc_history.append(acc)
        print(f"🎯 Accuracy on Task {task_id+1}: {acc:.2f}%")

    # --- Forgetting Calculation ---
    print("\n📉 Final Task Accuracies:")
    for i, acc in enumerate(acc_history):
        print(f"Task {i+1}: {acc:.2f}%")

    # --- Final Evaluation on Full CIFAR-100 Test Set ---
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    full_testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    full_test_loader = DataLoader(full_testset, batch_size=64, shuffle=False)
    print("\n🌍 Final Evaluation on Full CIFAR-100 Test Set:")
    total_acc = evaluate(model, full_test_loader)
    print(f"🧠 Total CIFAR-100 Accuracy: {total_acc:.2f}%")

if __name__ == "__main__":
    train()

