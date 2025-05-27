from uciml.datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# check which datasets can be imported: list_available_datasets()
# see also: ds.variables, ds.metadata.additional_info.summary

#ds_name = 'Heart Disease'
ds_name = "Contraceptive Method Choice"
X, y = load_dataset(ds_name)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.shortcut = nn.Linear(input_size, output_size)

    def forward(self, x):
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        shortcut = self.shortcut(x)
        return out3 + shortcut


def train_model(model, criterion, optimizer, n_epochs = 200, on_epoch_end=[]):
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        for f in on_epoch_end:
            f(epoch=epoch, model=model)

def save_pred_results_2cats(X, idx, dest):
    def f(epoch, model):
        model.eval()
        with torch.no_grad():
            outputs = model(X).squeeze()
            outputs = nn.functional.sigmoid(outputs)
            dest[epoch, idx, 1] = outputs
            dest[epoch, idx, 0] = 1 - outputs
    return f

n_epochs = 150
n_models = 100
n_cats = 2
res_train = np.full((n_models, n_epochs, len(X), n_cats), np.nan)
res_val = np.full((n_models, n_epochs, len(X), n_cats), np.nan)
res_test = np.full((n_models, n_epochs, len(X), n_cats), np.nan)

for seed in tqdm(range(n_models)):
    rng = np.random.default_rng(seed = seed)
    idxs = rng.permuted(np.arange(len(X)))
    # Proportions are set here
    i_train, i_val, i_test = np.split(idxs, [int(len(X) * .4), int(len(X) * .8)])
    X_train, X_val, X_test = X[i_train], X[i_val], X[i_test]
    y_train, y_val, y_test = y[i_train], y[i_val], y[i_test]

    torch.manual_seed(seed)
    model = MLP(
        input_size = X.shape[1],
        hidden_size = 32,  # Example hidden layer size
        output_size = y.shape[1]
    )

    train_model(
        model,
        criterion = nn.BCEWithLogitsLoss(),
        optimizer = optim.Rprop(model.parameters()),
        n_epochs = n_epochs,
        on_epoch_end=[
            save_pred_results_2cats(X_val, i_val, res_val[seed]),
            save_pred_results_2cats(X_train, i_train, res_train[seed]),
            save_pred_results_2cats(X_test, i_test, res_test[seed])
        ]
    )

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = nn.BCEWithLogitsLoss()(test_outputs, y_test)
        test_acc = (torch.round(nn.functional.sigmoid(test_outputs)) == y_test).mean(dtype=torch.float)

    print(f'model {seed:02d}, Test Loss: {test_loss.item():8.4f}, Test Acc.: {test_acc.item():02.4f}')

np.savez_compressed(
    f"{ds_name}_predictions.npz",
    predictions_validation=res_val,
    predictions_test=res_test,
    description=np.array(f"{ds_name} aggregate predictions in the shape; " \
                         "Models, Checkpoints, Predictions, Categories. " \
                         "Checkpoints where taken every epoch"),
    labels_validation = y.squeeze(),
    labels_test = y.squeeze()
)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

labels_val = y.squeeze().cpu().numpy()
labels_test = y.squeeze().cpu().numpy()
labels_train = y.squeeze().cpu().numpy()
val_scores = np.zeros((n_models, n_epochs))
test_scores = np.zeros((n_models, n_epochs))
train_scores = np.zeros((n_models, n_epochs))
X = np.arange(n_epochs)
for i in np.arange(n_models):
    preds_val = res_val[i]
    preds_test = res_test[i]
    preds_train = res_train[i]
    val_scores[i] = np.array([chkpnt_acc(chkpnt, labels_val) for chkpnt in preds_val])
    test_scores[i] = np.array([chkpnt_acc(chkpnt, labels_test) for chkpnt in preds_test])
    train_scores[i] = np.array([chkpnt_acc(chkpnt, labels_train) for chkpnt in preds_train])

i = .25
plt.title(f"Learning Curve ({ds_name})")
plt.plot(np.median(val_scores, axis=0), label="validation accuracy", color=colors[0])
plt.fill_between(X, np.quantile(val_scores, i, axis=0), np.quantile(val_scores, 1-i, axis=0), alpha=0.1, color=colors[0], linewidth=0)
plt.plot(np.median(test_scores, axis=0), label="test accuracy", color=colors[1])
plt.fill_between(X, np.quantile(test_scores, i, axis=0), np.quantile(test_scores, 1-i, axis=0), alpha=0.1, color=colors[1], linewidth=0)
plt.plot(np.median(train_scores, axis=0), label="train accuracy", color=colors[2])
plt.fill_between(X, np.quantile(train_scores, i, axis=0), np.quantile(train_scores, 1-i, axis=0), alpha=0.1, color=colors[2], linewidth=0)
plt.legend()
plt.savefig(f"{ds_name}_learning_curve.png")
plt.close()
