import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


def train( train_loader, device, model, decoder, m_optim,
          d_optim, epochs, criterion, mse, dataset_name ):
    
    lossi = []

    print(f"Dataset:{dataset_name}\nTraining ...")
    for epoch in tqdm(range(epochs)):

        LOSS = 0
        
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.unsqueeze(1).to(device).double(), labels.to(device)
    
            m_optim.zero_grad()
            d_optim.zero_grad()
    
            outs = model(inputs)
            caps_to_reconstruct = outs[torch.arange(outs.shape[0]), labels].unsqueeze(1)
            reconstructions = decoder(caps_to_reconstruct)
    
            margin_loss = criterion(outs, labels)
            mse_loss = mse(reconstructions, inputs)
            total_loss = margin_loss + 0.05 * mse_loss
            total_loss.backward()
    
            m_optim.step()
            d_optim.step()
    
            LOSS += total_loss.item()

        lossi.append(LOSS)

    plt.plot(lossi)
    plt.title(f"{dataset_name} Loss")



def evaluate_and_reconstruct( test_loader, device, model,
                              decoder, num_classes, limit,
                              dataset_name):
    
    model.eval()
    decoder.eval()

    _labels, _predictions, _reconstructions = [], [], {i:[] for i in range(num_classes)}
    correct, incorrect = 0, 0


    keep_checking = True
    
    def reconstructions_are_full( d=_reconstructions ):
        for recs in d.values():
            if len(recs) < limit:
                return False
        return True

    print(f"Dataset:{dataset_name}\nEvaluating ...")
    for inputs, labels in tqdm(test_loader):

        inputs, labels = inputs.unsqueeze(1).to(device).double(), labels.to(device)
        outs = model(inputs)
        caps_to_reconstruct = outs[torch.arange(outs.shape[0]), labels].unsqueeze(1)
        reconstructions = decoder(caps_to_reconstruct)

        preds = torch.sqrt( (outs**2).sum(-1) ).argmax(-1)
    
        
        _labels.extend(labels.tolist())
        _predictions.extend(preds.tolist())

        if keep_checking:
            if not reconstructions_are_full():
                for input, label, reconstruction in zip( inputs, labels, reconstructions ):
                        label = label.item()
                        if len(_reconstructions[label]) < limit:
                            _reconstructions[label].append((input,reconstruction))
            else:
                keep_checking = False
                    
    class_accuracies = []
    accuracy = accuracy_score(_labels, _predictions)
    cm = confusion_matrix(_labels, _predictions)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{dataset_name} : Confusion Matrix")
    plt.show()

    for cls, acc in enumerate(per_class_accuracy):
        class_accuracies.append((f"Class {cls}", acc))
    class_accuracies.append(("TOTAL", accuracy))

    # CLASSIFICATION ACCURACY
    df = pd.DataFrame(class_accuracies, columns=['Class', 'Accuracy'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    fig.suptitle(f"{dataset_name} : Class Accuracies")
    plt.show()
    
    # RECONSTRUCTIONS
    fig, axes = plt.subplots(num_classes, limit, figsize=(5*(limit+1), num_classes * 5))
    for cls, samples in _reconstructions.items():
        for i, package in enumerate(samples):
            input, reconstruction = package
            axes[cls, i].plot(input.squeeze().cpu().detach().numpy(), color='blue')
            axes[cls, i].plot(reconstruction.squeeze().cpu().detach().numpy(), color='orange')
    fig.suptitle(f"{dataset_name} : Reconstructions by Class")