import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

# Define the dataset
data = [
    (847876720719912960, 'abusive'),
    (850346419164553218, 'normal'),
    (847803697857953794, 'abusive'),
    # other data
    (51787273256050688, 'racism'),
    (551787717168619521, 'racism'),
    (551796768623435776, 'racism'),
    (551797843472556032, 'racism'),
    (551798472827883520, 'racism'),
    (551799120080293888, 'racism'),
    (551804052606967810, 'racism'),
    (551807282153934848, 'racism'),
    (551812016768172032, 'racism'),
    (551935344644349952, 'racism'),
    (551935446427504640, 'racism'),
    (551984700764332032, 'racism'),
    (552131027590586368, 'racism'),
    (552132765592727553, 'racism'),
    (552133306553073665, 'racism'),
    (552178267927883776, 'racism'),
    (552242912487305218, 'racism'),
    (552243885196734464, 'racism'),
    (552245992721252352, 'racism'),
    (552246388025982976, 'racism'),
    (552247283438587905, 'racism'),
    (552247950039326720, 'racism'),
    (552248563330482176, 'racism'),
    (552252662381432832, 'racism'),
    (552253146970337280, 'racism'),
    (552253495328247809, 'racism'),
    (552253987701788672, 'racism'),
    (552254351754817536, 'racism'),
    (552254776218365953, 'racism'),
    (552255235863756800, 'racism'),
    (552255665301749760, 'racism'),
    (552256116952805376, 'racism'),
    (552256504485527554, 'racism'),
    (552256837819449344, 'racism'),
    (552257230037213184, 'racism'),
    (552257644853862400, 'racism'),
    (552257867932106752, 'racism'),
    (552258072148582400, 'racism'),
    (552258490316492800, 'racism'),
    (552258840889028608, 'racism'),
    (552259230208499712, 'racism'),
    (552259592634114048, 'racism'),
    (552260022130864128, 'racism'),
    (552260502777102336, 'racism'),
    (552261139837370368, 'racism'),
    (552261703241453569, 'racism'),
    (552262508711399425, 'racism'),
    (552266284344553472, 'racism'),
    (552302969283436545, 'racism'),
    (552304521637285889, 'racism'),
    (552308801228259328, 'racism'),
    (552309137896640515, 'racism'),
    (552310296652828673, 'racism'),
    (552311142539079680, 'racism'),
    (552311743998070785, 'racism'),
    (552313304358846464, 'racism'),
    (552313552154157057, 'racism'),
    (552331392622010368, 'racism'),
    (552332311858262016, 'racism'),
    (552336365854412801, 'racism'),
    (552346465520349184, 'racism'),
    (552348770751770624, 'racism'),
    (552351551688564736, 'racism'),
    (552352216376692736, 'racism'),
    (552352897149976576, 'racism'),
    (552487055553757187, 'racism'),
    (552524577163988992, 'racism'),
    (552525214996004866, 'racism'),
    (552526460876242946, 'racism'),
    (552526654699237379, 'racism'),
    (552527064159768576, 'racism'),
    (552527629195427840, 'racism'),
    (552528092691181572, 'racism'),
    (552530221073969156, 'racism'),
    (552530506756423680, 'racism'),
    (552530948496318468, 'racism'),
    (552531880466456576, 'racism'),
    (552532181038686208, 'racism'),
    (552532522782171136, 'racism'),
    (552532883630723072, 'racism'),
    (552533207426797568, 'racism'),
    (552533544564953094, 'racism'),
    (552533736789913600, 'racism'),
    (552534024334614528, 'racism'),
    (552534253976969216, 'racism'),
    (552535646196494338, 'racism'),
    (552535998723547137, 'racism'),
    (552553410218557440, 'racism'),
    (552554260957634560, 'racism'),
    (552554758809321473, 'racism'),
    (552555113366446080, 'racism'),
    (552557234681430016, 'racism'),
    (552557686592516096, 'racism'),
    (552558144845406208, 'racism'),
    (552558562564534273, 'racism'),
    (552558873458913280, 'racism'),
    (552559358643417088, 'racism'),
    (552559570917134336, 'racism'),
    (552567397941796864, 'racism'),
    (552567745301454848, 'racism'),
    (552567939669716993, 'racism'),
    (552568185355239426, 'racism'),
    (552568707244118017, 'racism'),
    (552568932436295682, 'racism'),
    (552569179791187968, 'racism'),
    (552569711079485441, 'racism'),
    (552578641826029568, 'racism'),
    (552579242706235392, 'racism'),
    (552599309535834112, 'racism'),
    (552602629730078721, 'racism'),
    (552602854469275648, 'racism'),
    (552603067472814080, 'racism'),
    (552603232053100544, 'racism'),
    (552603982170820609, 'racism'),
    (552604209787318272, 'racism'),
    (552604397218193408, 'racism')
]

# Convert data to DataFrame
df = pd.DataFrame(data, columns=['id', 'label'])

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokens = tokenizer(list(df["label"]))

# Splitting data into tokens and labels
slicer = tokens["attention_mask"].sum(dim=1)
tokens = tokens["input_ids"]
labels = df['label'].apply(lambda x: 1 if x == 'abusive' else 0).values  # Encoding labels

# Define the CustomDataset
class CustomDataset(Dataset):
    def __init__(self, tokens, labels, slicer):
        self.tokens = tokens
        self.labels = labels
        self.slicer = slicer

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index], self.labels[index], self.slicer[index]

# Instantiate the dataset
dataset = CustomDataset(tokens, labels, slicer)

# Define the LightningDataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

# Initialize the LightningDataModule
data_module = CustomDataModule(dataset)
