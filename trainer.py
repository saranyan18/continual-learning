import torch
from torch import nn, optim
from dataset_loader import load_cifar100_cl_tasks
from cnn import ResNet18Backbone
from utils import create_filter_groups
from spectral_tracker import SpectralTracker
from grp_util_tracker import GroupUtilityTracker
from overlay_mod import OverlayModulator
from logger import create_log_entry

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
                    # Compute FIS scores and aggregate by group
                    filter_utilities = getattr(utility_tracker, 'filter_utilities', None)
                    if filter_utilities is None:
                        # If not tracked, fallback to group utility
                        group_util = utility_tracker.compute_group_utilities()
                        filter_utilities = {}
                        for (layer_name, idx), gid in overlay_mod.group_assignments.items():
                            filter_utilities[(layer_name, idx)] = group_util.get(gid, 0.0)
                    from utils import compute_fis_scores
                    fis_scores = compute_fis_scores(filter_flux, filter_utilities)
                    group_fis = {}
                    for (layer_name, idx), score in fis_scores.items():
                        gid = overlay_mod.group_assignments.get((layer_name, idx))
                        if gid is not None:
                            group_fis.setdefault(gid, []).append(score)
                    group_fis = {gid: sum(scores)/len(scores) for gid, scores in group_fis.items()}
                    overlay_mod.compute_overlay_weights(group_flux=None, group_utility=group_fis)
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
    num_tasks = 5
    batch_size = 16
    epochs = 10 # You can increase later

    # --- Load Data ---
    tasks = load_cifar100_cl_tasks(num_tasks=num_tasks, batch_size=batch_size)

    # --- Initialize Model + Trackers ---
    model = ResNet18Backbone(num_classes=100, pretrained=False).to(device)
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

    # --- Logging Experiment ---
    create_log_entry(
        model_name="ResNet18Backbone",
        architecture=str(model),
        num_tasks=num_tasks,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name="Adam",
        learning_rate=1e-3,
        modulation="OverlayModulator",
        cosine_scaling=getattr(model.classifier, 'scale', None).item() if hasattr(model, 'classifier') and hasattr(model.classifier, 'scale') else None,
        task_accuracies=acc_history,
        full_accuracy=total_acc,
        notes="Final evaluation and logging."
    )

if __name__ == "__main__":
    train()

