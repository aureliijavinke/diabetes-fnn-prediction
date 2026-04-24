import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)

# 1. Reproducējamība
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 2. stepbystep klase
class StepByStep:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = None
        self.val_loader = None
        self.losses = []
        self.val_losses = []

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_step(self, x_batch, y_batch):
        self.model.train()
        yhat = self.model(x_batch)
        loss = self.loss_fn(yhat, y_batch)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def val_step(self, x_batch, y_batch):
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(x_batch)
            loss = self.loss_fn(yhat, y_batch)
        return loss.item()

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            mini_batch_losses = []

            for x_batch, y_batch in self.train_loader:
                loss = self.train_step(x_batch, y_batch)
                mini_batch_losses.append(loss)

            train_loss = np.mean(mini_batch_losses)
            self.losses.append(train_loss)

            if self.val_loader is not None:
                val_batch_losses = []
                for x_batch, y_batch in self.val_loader:
                    val_loss = self.val_step(x_batch, y_batch)
                    val_batch_losses.append(val_loss)

                validation_loss = np.mean(val_batch_losses)
                self.val_losses.append(validation_loss)

            if (epoch + 1) % 50 == 0:
                if self.val_loader is not None:
                    print(
                        f"Epoch {epoch+1:4d} | "
                        f"train_loss = {train_loss:.4f} | "
                        f"val_loss = {validation_loss:.4f}"
                    )
                else:
                    print(f"Epoch {epoch+1:4d} | train_loss = {train_loss:.4f}")

    def predict_proba(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        return probs

# 3. ieladejam datus..
# fails atrodas:
df = df = pd.read_csv("data/diabetes.csv")

print("Datu izmērs:", df.shape)
print(df.head())

# 4. sagatavojam musu datus
# jusu dataset ir vairākās kolonnās "0", tapec es aizstāšu ar mediānu, lai modelis netiktu sabojāts.
cols_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in cols_with_invalid_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Ievades un mērķis
X = df[
    [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
].values

y = df["Outcome"].values

# Train / Validation sadalījums
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=.3, random_state=13, stratify=y
)

# Standartizācija
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# 5. tensori
x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()


# 6. Dataset/dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False
)

# 7. FNN modelis
# 8 ievades parametri -> slēptie slāņi -> 1 izeja
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)

# 8. Treninsh
n_epochs = 300
sbs.train(n_epochs)

# 9. Prognozes
probs = sbs.predict_proba(x_val_tensor).numpy().ravel()
preds = (probs >= 0.5).astype(int)

# 10. Confusion matrix
cm = confusion_matrix(y_val, preds)
tn, fp, fn, tp = cm.ravel()

print("\n========== REZULTĀTI ==========")
print("Confusion Matrix:")
print(cm)

# 11. precizitate
accuracy = accuracy_score(y_val, preds)
print(f"Accuracy = {accuracy:.4f}")


# 12. precision/recall?
precision = precision_score(y_val, preds, zero_division=0)
recall = recall_score(y_val, preds, zero_division=0)

print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")

# 13. true/false positive reiti
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print(f"True Positive Rate  = {tpr:.4f}")
print(f"False Positive Rate = {fpr:.4f}")

# 14. roc auc
roc_auc = roc_auc_score(y_val, probs)
print(f"ROC AUC = {roc_auc:.4f}")

# ROC punkti
fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_val, probs)
roc_curve_auc = auc(fpr_curve, tpr_curve)

print(f"AUC from roc_curve = {roc_curve_auc:.4f}")

# 15. papildu izvads
print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)


# 16. rezulati
print("\n========== INTERPRETĀCIJA ==========")
print(
    "Confusion Matrix parāda, cik daudz pacientu modelis klasificēja pareizi un cik kļūdaini. "
    "TN ir pareizi atpazīti pacienti bez diabēta, TP ir pareizi atpazīti pacienti ar diabētu. "
    "FP nozīmē, ka modelis kļūdaini atzīmēja pacientu kā diabēta slimnieku, bet FN nozīmē, ka "
    "modelis neatrada īstu diabēta gadījumu. Accuracy rāda kopējo pareizo klasifikāciju īpatsvaru. "
    "Precision parāda, cik uzticamas ir pozitīvās prognozes, savukārt Recall rāda, cik lielu daļu "
    "no visiem patiesajiem diabēta gadījumiem modelis atrada. TPR ir tas pats, kas Recall, bet FPR "
    "parāda kļūdaini pozitīvo klasifikāciju daļu starp patiesi negatīvajiem pacientiem. ROC AUC raksturo "
    "modeļa spēju atšķirt abas klases dažādiem sliekšņiem; jo tuvāk 1.0, jo labāks modelis."
)

# 17. roc grafiks
fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_val, probs)
roc_curve_auc = auc(fpr_curve, tpr_curve)

plt.figure(figsize=(8,6))

plt.plot(
    fpr_curve,
    tpr_curve,
    linewidth=2,
    label=f"ROC līkne (AUC = {roc_curve_auc:.4f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=1,
    label="Nejaušs klasifikators"
)

plt.xlabel("Viltus pozitīvo gadījumu īpatsvars (FPR)")
plt.ylabel("Patieso pozitīvo gadījumu īpatsvars (TPR)")
plt.title("ROC līkne diabēta prognozēšanas modelim")

plt.legend(loc="lower right")
plt.grid(True)

plt.show()
