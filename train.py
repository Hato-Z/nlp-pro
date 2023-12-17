import torch
import torch.nn as nn
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
import logging

from transformer_basic import *
from torchtext.data.metrics import bleu_score

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

STEPS=0

lr_step=[]

class NoamOpt:
    def __init__(self, model_size, factor=2, warmup=1000):
        # self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train(model, iterator, optimizer, criterion, clip, scheduler):
    model.train()
    
    epoch_loss = 0
    
    global STEPS
    global lr_step
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        
        STEPS += 1
        lr = scheduler.rate(STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        
        
    
    tem=0
    for param_group in optimizer.param_groups:
            tem=param_group['lr']
    print(tem)
    print(STEPS)
    lr_step.append(tem)
    
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    


    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):

    trgs = []
    pred_trgs = []

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)

def main():
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG), root='data')

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=True)
    
    # for i, batch in enumerate(train_iterator):
    #     print(batch.src)
    #     print(batch.trg)
    #     if i >3:
    #         break
        

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    model.apply(initialize_weights)

    # optimizer
    # using fixed learning rate by BERT
    # but original Transformer has a "warm-up" and then a "cool-down" period
    LEARNING_RATE = 0.0002
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # loss func, ignore <pad> tokens
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    print("main start")

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    # logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    train_loss_arr=[]
    val_loss_arr=[]
    train_ppl_arr=[]
    val_ppl_arr=[]
    
    noam = NoamOpt(model_size=HID_DIM, factor=0.5, warmup=1000)

    for epoch in range(N_EPOCHS):
        
     
            
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, noam)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')

        train_loss_arr.append(train_loss)
        train_ppl_arr.append(math.exp(train_loss))
        val_loss_arr.append(valid_loss)
        val_ppl_arr.append(math.exp(valid_loss))

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

    print(f'BLEU score = {bleu_score*100:.2f}')
    logger.info(f'\t BLEU score = {bleu_score*100:.2f}')

    x = range(N_EPOCHS)
    
    plt.figure(1)
    plt.plot(x, train_loss_arr, label='Train loss')
    plt.plot(x, val_loss_arr, label='Val loss')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
    
    plt.figure(2)
    plt.plot(x, train_ppl_arr, label='Train PPL')
    plt.plot(x, val_ppl_arr, label='Val PPL')
    plt.title('PPL')
    plt.xlabel('epoch')
    plt.ylabel('PPL')
    plt.legend()
    plt.savefig('PPL.png')
    plt.show()
    
    plt.figure(3)
    plt.plot(x, lr_step, label='lrstep')
    plt.title('lr')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.legend()
    plt.savefig('lrstep.png')
    plt.show()

# model.load_state_dict(torch.load('tut6-model.pt'))
#
# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
 
if __name__ == '__main__':
    main()
