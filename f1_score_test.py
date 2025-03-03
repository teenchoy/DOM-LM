from pathlib import Path
import src.preprocess as preprocess
from src.domlm import DOMLMConfig
import numpy
import torch
from src.domlm.modeling_domlm import DOMLMForTokenClassification
from src.utils import label2id
from sklearn.metrics import f1_score


def getLabelInfo(labelfiles):
    labelinfo = {}
    for labelfile in labelfiles:
        label = labelfile.name.replace('.txt','').split('-')[-1]
        lines = open(labelfile, "r").readlines()
        for line in lines[2:]:
            lineTokens = line.split('\t')
            page_id = lineTokens[0]
            if page_id not in labelinfo:
                labelinfo[page_id] = {}
            nums = lineTokens[1]
            value = lineTokens[2].strip()
            labelinfo[page_id][label] = {
                "nums": nums,
                "value": value
            }

    return labelinfo

config = DOMLMConfig.from_json_file('/content/DOM-LM/domlm-config/config.json')
config.num_labels = len(label2id.keys()) + 1
print('1234a')
model = DOMLMForTokenClassification.from_pretrained("/content/drive/MyDrive/colab/ae_trained_output/checkpoint-298", config=config)
# model = DOMLMForSequenceClassification.from_pretrained("/Users/zehengxiao/Downloads/checkpoint-2955", config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

import pickle
pkl = Path("/content/output/movie/movie-allmovie(2000)").glob("1999.pkl")
features = ""
with open(sorted(pkl)[0],'rb') as f:
    features = pickle.load(f) 
batch = {k: torch.tensor([f[k] for f in features]) for k in features[0].keys()}
labels = batch['labels']
labels = labels.apply_(lambda x: 8 if -100 else x)
del batch['labels']
# optimizer.zero_grad()
outputs = model.domlm(**batch)
logits = model.classifier(outputs[0])
# logits.shape  
cc = torch.nn.CrossEntropyLoss()
loss = cc(logits.view(-1, config.num_labels), labels.view(-1))
# loss.backward()
# optimizer.step()
print(f'{loss=}')
# loss
# model.eval()
a = numpy.array(logits.argmax(dim=-1).tolist())
b = numpy.array(labels.tolist())
score = f1_score(a.flatten(), b.flatten(), average="macro")
print("F1 Score:", score)
# model.train()

batch2 = {k: torch.tensor([f[k] for f in features]) for k in features[0].keys()}
outputss = model(**batch2)
loss2 = outputss[0]
print(f" model loss: {loss2}")