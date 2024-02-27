import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from transformers import BertTokenizer, BertModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
import argparse

class FramingDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['text']
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)



class BERTClass(torch.nn.Module):
    def __init__(self, model):
        super(BERTClass, self).__init__()
        if model == 'bert':
            self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        elif model == 'roberta':
            self.bert_model = RobertaModel.from_pretrained('roberta-base')
        elif model == 'mbert':
            self.bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased', return_dict=True)
        elif model == 'XLM-roberta':
            self.bert_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, len(target_list))

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train_model(n_epochs, training_loader, validation_loader, model, optimizer, checkpoint_path, best_model_path):
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        print(training_loader)
        for batch_idx, data in enumerate(training_loader):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print('############# Epoch {}: Training End     #############'.format(epoch))

        print('############# Epoch {}: Validation Start   #############'.format(epoch))
        ######################
        # validate the model #
        ######################

        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                ids = data['input_ids'].to(device, dtype = torch.long)
                mask = data['attention_mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            print('############# Epoch {}: Validation End     #############'.format(epoch))

            train_loss = train_loss/len(training_loader)
            valid_loss = valid_loss/len(validation_loader)
            print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
                ))

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            checkpoint_path = checkpoint_path+str(checkpoint['epoch'])+'.pt'
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss

            print('############# Epoch {}  Done   #############\n'.format(epoch))

    return model


if __name__ == "__main__":
    import time
    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type = int, default = 10, help = "insert the number of epochs. default 10")
    parser.add_argument("--lr", type = float, default = 1e-5, help = "insert the LEARNING_RATE. default 1e-5")
    parser.add_argument("--model", type = str, default = 'bert', help = "insert the model name. default bert")
    parser.add_argument("--train_path", type = str, default = 'data/train0.csv', help = "insert the training dataset filename. default train0.csv")
    parser.add_argument("--test_path", type = str, default = 'data/test0.csv', help = "insert the testing dataset filename. default test0.csv")
    parser.add_argument("--batch_size", type = int, default = 32, help = "insert the batch size. default 32")
    parser.add_argument("--output_file", type = bool, default = False, help = "generate test prediction file. default False")
    parser.add_argument("--output_filename", type = str, default = 'predictions.csv', help = "name of the output predictions filename. default predictions.csv")

    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    train_df = train_df.drop(columns = 'language')

    target_list = list(train_df.columns)
    target_list.pop(0)

    # hyperparameters
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = args.batch_size
    VALID_BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.model == 'mbert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    elif args.model == 'XLM-roberta':
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')


    train_size = 0.9
    train_df0 = train_df.sample(frac=train_size, random_state=200).reset_index(drop=True)
    val_df = train_df.drop(train_df0.index).reset_index(drop=True)


    train_dataset = FramingDataset(train_df0, tokenizer, MAX_LEN)
    valid_dataset = FramingDataset(val_df, tokenizer, MAX_LEN)


    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

    val_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BERTClass(model = args.model)
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    val_targets=[]
    val_outputs=[]

    ckpt_path = "curr_ckpt/"
    best_model_path = "best_model_checkpts/epoch"+str(EPOCHS)+".pt"


    model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, ckpt_path, best_model_path)


#  ***********************************Testing*********************************
    test_df = pd.read_csv(args.test_path)
    columns_list = list(test_df.columns)

    test_df1 = test_df.drop(columns = ['language'], axis='columns')

    test_df_out = test_df.drop(target_list, axis = 'columns').values

    test_dataset = FramingDataset(test_df1, tokenizer, MAX_LEN)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)

    predictions, labels = [],[]

    model.eval()

    with torch.no_grad():
        for idx, encodings in enumerate(test_dataset):
            input_ids = encodings['input_ids'].unsqueeze(0).to(device, dtype=torch.long)
            attention_mask = encodings['attention_mask'].unsqueeze(0).to(device, dtype=torch.long)
            token_type_ids = encodings['token_type_ids'].unsqueeze(0).to(device, dtype=torch.long)
            targets = encodings['targets'].to(device, dtype = torch.float)
            output = model(input_ids, attention_mask, token_type_ids)
            output = torch.sigmoid(output).cpu().detach().numpy()
            predictions.append(output.flatten())
            labels.append(targets.cpu().detach().numpy())
            if idx == len(test_dataset)-1:
                break

    predictions = np.array(predictions)
    labels = np.array(labels)

    predictions2 = []
    for p in predictions:
        l = np.zeros(len(target_list))
        l[np.argmax(p)] = 1
        predictions2.append(l)

    predictions2 = np.array(predictions2)

    if args.output_file:
        preds_out = np.concatenate((test_df_out, predictions2), axis=1)
        preds_out_df = pd.DataFrame(preds_out, columns = columns_list)
        preds_out_df.to_csv(args.output_filename, index = False)

    labels2 = np.argmax(labels, axis=1)
    predictions3 = np.argmax(predictions2, axis=1)

    print("*************************************************************************************")
    print(args)
    print("*************************************************************************************")
    print("Mean Accuracy, \n", accuracy_score(predictions3, labels2))
    print("*************************************************************************************")
    print("Classification Report\n", classification_report(labels, predictions2, target_names=target_list, zero_division=0))
    print("*************************************************************************************")
    print("Total runtime ", (time.time()-start)/60)
    print("*************************************************************************************")
