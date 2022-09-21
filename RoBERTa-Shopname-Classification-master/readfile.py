import pandas as pd
from torch.utils.data import Dataset, DataLoader
import re


def read_text_from_tsv(filename,train_rate=0.8):
    with open(filename,encoding='gb18030',errors='ignore') as f:
        pdframe = pd.read_csv(f, sep='\t')
        pdframe.drop(columns=['DATA_MONTH', 'ITEM_ID', 'BRAND_ID', 'BRAND_NAME',
                              'ITEM_PRICE', 'ITEM_SALES_VOLUME', 'ITEM_SALES_AMOUNT',
                              'ITEM_FAV_NUM', 'TOTAL_EVAL_NUM', 'ITEM_STOCK', 'ITEM_DELIVERY_PLACE',
                              'ITEM_PROD_PLACE', 'ITEM_PARAM', 'USER_ID', 'SHOP_NAME'], inplace=True)
        pdframe = pdframe[~(pdframe['CATE_NAME_LV1'].isnull())]
        classes = pdframe['CATE_NAME_LV1'].unique()
        tag2id = {clas: ind for ind, clas in enumerate(classes)}
        pdframe['num_class'] = pdframe['CATE_NAME_LV1'].map(tag2id)
        trainx = pdframe.iloc[0:int(len(pdframe)*train_rate),0]
        trainy = pdframe.iloc[0:int(len(pdframe)*train_rate),-1]
        textx = pdframe.iloc[int(len(pdframe)*train_rate):len(pdframe),0]
        texty = pdframe.iloc[int(len(pdframe)*train_rate):len(pdframe),-1]
        return trainx, trainy, textx, texty, tag2id


class NERset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.datasize = len(dataset)

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        return self.dataset[index]


def sers2set(trainx,trainy,testx,testy):
    tax = list(trainx)
    tay = list(trainy)
    tex = list(testx)
    tey = list(testy)
    traindata = [(a,b) for a,b in zip(tax,tay)]
    testdata = [(a,b) for a,b in zip(tex,tey)]
    trainset = NERset(traindata)
    testset = NERset(testdata)
    return trainset,testset


def coffate_fn(examples):
    sents = []
    tags = []
    for itemname,tag in examples:
        itemname = re.sub('[^\u4e00-\u9fa5]+','',itemname)
        sents.append(itemname)
        tags.append(tag)
    tokenized_inputs = tokenizer(sents,
                                 truncation=True,
                                 padding=True,
                                 # return_offsets_mapping=True,
                                 is_split_into_words=False,
                                 max_length=100,
                                 return_tensors="pt")
    targets = torch.tensor(tags)
    return tokenized_inputs, targets


def loadset(filename,train_rate=0.8,batch_size=16):
    trainx, trainy, textx, texty, tag2id = read_text_from_tsv(filename, train_rate)
    trainset,testset = sers2set(trainx, trainy, textx, texty)
    train_dataloader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  collate_fn=coffate_fn,
                                  shuffle=True)
    test_dataloader = DataLoader(testset,
                                 batch_size=batch_size,
                                 collate_fn=coffate_fn,
                                 shuffle=True)
    return train_dataloader, test_dataloader, tag2id