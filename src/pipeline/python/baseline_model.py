import pandas as pd
import sklearn as sk
import numpy as np
import random
#Purpose of model: Baseline model to predict the natural processing of an epitope inside the cell.
# MHCI pathway: predict whether an epitope will be processed and presented on the cell surface by MHCI molecules.
# Data: Eluted ligand data, with added features such as length, hydrophobicity, flanking regions, etc.

#Reading the data
df = pd.read_csv("data/processed/epitopes_with_features.csv")

features = [
    "pep_length",
    "peptide_hydrophobic_frac",
    "peptide_charge_total"
]

# positives
pos_df = df[features].copy()
pos_df["label"] = 1

def generate_negative_peptides(df, n_neg_per_pos=1):
    
    positives = set(df["peptide"])
    negatives = []

    for _, row in df.iterrows():
        
        seq = row["sequence"]
        length = row["pep_length"]
        
        for _ in range(n_neg_per_pos):
            
            if len(seq) <= length:
                continue
            
            start = random.randint(0, len(seq) - length)
            pep = seq[start:start+length]
            
            if pep not in positives:
                negatives.append({
                    "peptide": pep,
                    "sequence": seq,
                    "pep_length": length
                })
                
    neg_df = pd.DataFrame(negatives)
    return neg_df

# Generate negative samples
neg_df = generate_negative_peptides(df, n_neg_per_pos=1)
print("Generated negative samples:", neg_df.shape)

# Adding featues to negative samples
hydrophobicity = {
    "A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,
    "G":-0.4,"H":-3.2,"I":4.5,"K":-3.9,"L":3.8,
    "M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,
    "S":-0.8,"T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3
}

charge = {
    "A":0,"C":0,"D":-1,"E":-1,"F":0,
    "G":0,"H":0,"I":0,"K":1,"L":0,
    "M":0,"N":0,"P":0,"Q":0,"R":1,
    "S":0,"T":0,"V":0,"W":0,"Y":0
}
hydrophobic_residues = set(["A","I","L","M","F","V","W","Y"])
def calculate_peptide_properties(peptide):
    
    if peptide is None or peptide == "":
        return np.nan, np.nan
    
    aa_list = list(peptide)
    n = len(aa_list)

    charge_total = sum(charge.get(aa, 0) for aa in aa_list)
    
    hydrophobic_frac = sum(aa in hydrophobic_residues for aa in aa_list) / n

    return hydrophobic_frac, charge_total

neg_df["peptide_hydrophobic_frac"], neg_df["peptide_charge_total"] = zip(
    *neg_df["peptide"].apply(calculate_peptide_properties)
)

neg_df["label"] = 0

full_df = pd.concat([pos_df, neg_df], ignore_index=True)

#dimension of full df is (number of samples, number of features + 1 for label)
print(full_df.shape)
print(full_df[features].isna().sum())

#Baseline model: predict the most common class
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


X = full_df[features]
print(X)
y = full_df["label"]
print(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", model.score(X_test, y_test))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

#What does the model predict? 
np.mean(model.predict(X_test))
