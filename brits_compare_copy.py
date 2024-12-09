#from torch.utils.data import Dataset
from typing import Literal
import numpy as np
from sklearn import metrics
# from imblearn.metrics import specificity_score
# from utils import makedir

#Model 
import torch
import torch.nn as nn                     # general structure of a net 
import torch.nn.functional as F           # for Relu
from torch.autograd import Variable
#from torch.utils import data
import torch.optim as optim

from src.methods.brits.lightningmodule import BRITSLightningModule
from src.datamodules.daicwoz import DAICWOZDatamodule
from src.methods.brits.modules import MultiTaskBRITS
import wandb
import copy
from tqdm import trange

from multitask_missing_q10_ours_copy import Model_brits_att, to_var, get_loader, binary_cross_entropy_with_logits, TemporalDecay, FeatureRegression
from lightning.pytorch import seed_everything
from typing import Dict, Any

#--- Run RITS model with Attentions
class Model_rits_att(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(Model_rits_att, self).__init__()
        self.config = config
        self.build()
        
        # Attention following AudiBert 
        self.W_s1 = nn.Linear(self.config.rnn_hid_size, 350)
        self.W_s2 = nn.Linear(350, 30)
    def build(self):
        if self.config.rnn_name=='LSTM':
            self.rnn_cell = nn.LSTMCell(self.config.n_series * 2, self.config.rnn_hid_size)
        elif self.config.rnn_name=='GRU':
            self.rnn_cell = nn.GRUCell(self.config.n_series * 2, self.config.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = self.config.n_series, output_size = self.config.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.config.n_series, output_size = self.config.n_series, diag = True)

        self.hist_reg = nn.Linear(self.config.rnn_hid_size, self.config.n_series)
        self.feat_reg = FeatureRegression(self.config.n_series)

        self.weight_combine = nn.Linear(self.config.n_series * 2, self.config.n_series)

        self.dropout = nn.Dropout(p = 0.25)
        #self.out = nn.Linear(self.config.rnn_hid_size, 1)
        self.out = nn.Linear(self.config.rnn_hid_size*30, 1)
        
    def attention_rnn(self, rnn_output):
        #attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(rnn_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        #print('values.shape:',values.shape)          #torch.Size([1, 40, 12])
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        
        # to store historical hidden size from rnn
        H_rnn = torch.zeros(values.shape[0], self.config.seq_len, self.config.rnn_hid_size)  # (batch, sequence, hiden_dize)

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.config.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.config.rnn_hid_size)))

        #if torch.cuda.is_available():
        #    h, c = h.cuda(), c.cuda()
        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(self.config.seq_len):
            x = values[:, t, :]
            #print('x.shape:',x.shape)     # x.shape: torch.Size([1, 12])
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)
            if self.config.rnn_name=='LSTM':
                h, c = self.rnn_cell(inputs, (h, c))         # h lstm: torch.Size([1, 32]
                #print('h lstm:', h.shape)  
            elif self.config.rnn_name=='GRU':
                h = self.rnn_cell(inputs, h)                 # h GRU: torch.Size([1, 32]
                #print('h GRU:', h.shape)
            H_rnn[:,t,:] = h
            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)
        
        # Attentions 
        attn_weight_matrix = self.attention_rnn(H_rnn)
        hidden_matrix = torch.bmm(attn_weight_matrix, H_rnn)
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
        #print('attention_output.shape:',attention_output.shape)
        #print('h.shape:',h.shape)

        #y_h = self.out(h)
        y_h = self.out(attention_output)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss / self.config.seq_len + y_loss * 0.1, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

#--- Run BRITS model with Attentions 
class Model_brits_att(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(Model_brits_att, self).__init__()
        self.config = config
        self.build()

    def build(self):
        self.rits_f = Model_rits_att(config = self.config)
        self.rits_b = Model_rits_att(config = self.config)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            #if torch.cuda.is_available():
            #    indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret


def format_batch_for_ricardo(batch, model):
    params_type = next(iter(model.parameters()))
    data_ric = copy.deepcopy(batch)

    new_batch = {
        'forward': {
            'values': batch['X'].type_as(params_type),
            'masks': batch['missing_mask'].type_as(params_type),
            'deltas': batch['deltas'].type_as(params_type),
            'evals': batch['X_ori'].type_as(params_type),
            'eval_masks': batch['indicating_mask'].type_as(params_type),
        },
        'backward': {
            'values': batch['back_X'].type_as(params_type),
            'masks': batch['back_missing_mask'].type_as(params_type),
            'deltas': batch['back_deltas'].type_as(params_type),
            'evals': batch['X_ori'].flip(1).type_as(params_type),
            'eval_masks': batch['indicating_mask'].flip(1).type_as(params_type),
        },
        'is_train': torch.ones(len(batch['X'])),
        'labels': batch['label'].long(),
    }

    data_ric = to_var(new_batch)
    return data_ric

def evaluate(model, val_iter, model_choice: Literal["kevin", "ricardo"] = "kevin"):
    model.eval()
    labels = []
    preds = []
    loss = []

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        is_train = torch.zeros(len(data['label']))
        if model_choice == "kevin":
            ret = model.validation_step(data, idx)
            loss.append(ret['loss'].data.cpu().numpy())
            pred = ret['clf_logits'][:, -1].data.cpu().numpy()
            label = data['label'].data.cpu().numpy()
            eval_masks = data['indicating_mask'].data.cpu().numpy()
            eval_ = data['X_ori'].data.cpu().numpy()
            imputation = ret['imputed_data'].data.cpu().numpy()
        else:
            data = format_batch_for_ricardo(data, model)
            ret = model.run_on_batch(data, None)
            loss.append(ret['loss'].data.cpu().numpy())
            pred = ret['predictions'].data.cpu().numpy()
            label = ret['labels'].data.cpu().numpy()
            eval_masks = ret['eval_masks'].data.cpu().numpy()
            eval_ = ret['evals'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32') 
    preds = np.asarray(preds)
    dummy_pred = np.zeros_like(preds)
    dummy_pred[preds >= 0.5] = 1

    AUC = metrics.roc_auc_score(labels, preds)
    BA = metrics.balanced_accuracy_score(labels,dummy_pred)
    F1 = metrics.f1_score(labels,dummy_pred)
    Recall = metrics.recall_score(labels,dummy_pred)
    # SF = specificity_score(labels,dummy_pred)

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    #print('imputations:', imputations)

    if len(imputations)!=0:  
        MAE = np.abs(evals - imputations).mean()
        MRE = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    else: # when the missing ration is setting to zero
        MAE = 0
        MRE = 0
    
    #print ('MAE', np.abs(evals - imputations).mean())
    # return MAE, MRE, AUC, BA, F1, Recall,
    return {
        "MAE": MAE,
        "MRE": MRE,
        "AUC": AUC,
        "BA": BA,
        "F1": F1,
        "Recall": Recall,
        "loss": np.mean(loss),
    }


def train():
    wandb.init(project="multitask_missing")
    seed_everything(777)
    
    config = wandb.config

    if config.open_face == 'action_unit':
        config.n_series = 14  # 14, for action unit
    elif config.open_face == 'eye_gaze':
        config.n_series = 12
    elif config.open_face == 'landmark':
        config.n_series = 136
    elif config.open_face == 'all':
        config.n_series = (14 + 12 + 136)

    dm = DAICWOZDatamodule(
        label='phq8',
        open_face=config.open_face,
        question=config.question,
        delta_steps=1,
        delta_average=False,
        regen=True,
        ratio_missing=config.ratio_missing,
        type_missing=config.type_missing,
        val_ratio=0.0, # dummy
        batch_size=config.batch_size,
        num_workers=1,
        ricardo=True #ensures we get train set of all is_train == 1 and val set of all is_train == 0
)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if config.model_version == "kevin":
        model = BRITSLightningModule(
            rnn_hidden_size=config.rnn_hid_size,
            reconstruction_weight=config.reconstruction_weight,
            classification_weight=config.classification_weight,
            question=config.question,
            pypots=False,
            lr=config.lr,
            n_features=config.n_series,
            n_classes=2,
        )
        data_dict = {
            'n_steps': config.seq_len,
            'n_features': config.n_series,
            'n_classes': 2,
            'rnn_hidden_size': config.rnn_hid_size,
            "classification_weight": config.classification_weight,
            "reconstruction_weight": config.reconstruction_weight
        }
        model.model = MultiTaskBRITS(**data_dict)

    else:
        model = Model_brits_att(config=config)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    scores = {
        "MAE": [],
        "MRE": [],
        "AUC": [],
        "BA": [],
        "F1": [],
        "Recall": [],
    }
    
    for epoch in trange(config.epochs, desc=f"{config.question}-Epoch"):
        model.train()
        
        for idx, data in enumerate(train_loader):
            if config.model_version == "ricardo":
                data = format_batch_for_ricardo(data, model)
                ret_ric = model.run_on_batch(data, optimizer=optimizer)
                wandb.log({
                    "train/loss": ret_ric['loss'].data.cpu().numpy()
                })

            else:
                optimizer.zero_grad()
                return_kev = model.training_step(data, idx)
                loss_kev = return_kev['loss']
                loss_kev.backward()
                optimizer.step()

                wandb.log({
                    "train/loss": loss_kev.data.cpu().numpy()
                })
                    
           
            eval_out = evaluate(model, val_loader, model_choice=config.model_version)
            scores['F1'].append(eval_out['F1'])
            scores['MAE'].append(eval_out['MAE'])
            scores['MRE'].append(eval_out['MRE'])
            scores['AUC'].append(eval_out['AUC'])
            scores['BA'].append(eval_out['BA'])
            scores['Recall'].append(eval_out['Recall'])

    for key, value in scores.items():
        sorted_values = sorted(value, reverse=True)
        highest_five = sorted_values[:5]
        wandb.log({f"val/{key}": np.mean(highest_five)})


    wandb.finish()

if __name__ == '__main__':
    train()


# if __name__ == '__main__':

#     for question in [
#         # 'advice_yourself',
#         # 'anything_regret',
#         # 'argued_someone',
#         # 'controlling_temper',
#         # 'doing_today',
#         'dream_job'
#     ]:
#         seed_everything(42)
#         SEQ_LEN = 40                           # number of period in the ts, t = 1, 2, 3, 4, 5.
#         RNN_HID_SIZE = 32                     # hidden node of the rnn 
#         batch_size = 32
#         model_name = 'BRITS_ATT' # RITS
#         question = question
#         open_face = 'eye_gaze'
#         epochs = 100
#         #N_SERIES = 12                          # number of series Rd, 12:eye, 136:landmark,14:action unit
#         lr = 1e-3
#         repetitions = 1
#         ratio_missing = 0.05
#         type_missing = 'Random' # CMV
#         rnn_name = 'LSTM' # GRU
#         experiment_name = 'exp01'
#         use_ricardo_loader = True
        
        
#         if open_face=='action_unit':
#             N_SERIES = 14  # 14, for action unit
#         elif open_face=='eye_gaze':
#             N_SERIES = 12
#         elif open_face=='landmark':
#             N_SERIES = 136
#         elif open_face=='all':
#             N_SERIES = (14+12+136)
        
#         hyperparameters = {
#         "question": question,
#         "open_face": open_face,
#         "epochs": epochs,
#         "lr": lr,
#         "ratio_miss": ratio_missing,
#         "type_missing": type_missing,
#         "rnn_name": rnn_name,
#         "batch_size": batch_size,
#         "model_name": model_name
#     }
#         wandb.init(
#             project="multitask_missing", 
#             name=f"brits_compare_question={question}_open_face={open_face}",
#             config=hyperparameters)

        

#         dm = DAICWOZDatamodule(
#             label='phq8',
#             open_face=open_face,
#             question=question,
#             delta_steps=1,
#             delta_average=False,
#             regen=True,
#             ratio_missing=ratio_missing,
#             type_missing=type_missing,
#             val_ratio=0.0, # dummy
#             batch_size=batch_size,
#             num_workers=0,
#             ricardo=True #ensures we get train set of all is_train == 1 and val set of all is_train == 0
#         )
#         dm.prepare_data()
#         dm.setup()
#         train_loader = dm.train_dataloader()
#         val_loader = dm.val_dataloader()

#         # ric_loader = get_loader(
#         #     question=question,
#         #     open_face=open_face,
#         #     batch_size=batch_size,
#         #     ratio_missing=ratio_missing,
#         #     type_missing=type_missing,
#         #     shuffle=False
#         # )
#         model_ric = Model_brits_att(rnn_name)
#         optimizer_ric = optim.Adam(model_ric.parameters(), lr = lr) # switch optimizer?? adamW
        
#         model_kev = BRITSLightningModule(
#                 rnn_hidden_size=RNN_HID_SIZE,
#                 reconstruction_weight=1.0,
#                 classification_weight=0.1,
#                 question=question,
#                 pypots=False,
#                 lr=lr,
#                 n_features = N_SERIES,
#                 n_classes=2,
#         )
#         data_dict = {
#             'n_steps': SEQ_LEN,
#             'n_features': N_SERIES,
#             'n_classes': 2,
#             'rnn_hidden_size': RNN_HID_SIZE,
#             "classification_weight": 0.1,
#             "reconstruction_weight": 1.0
#         }
#         model_kev.model = MultiTaskBRITS(**data_dict)
#         optimizer_kev = optim.Adam(model_kev.parameters(), lr = lr)

#         for epoch in trange(epochs, desc=f"{question}-Epoch"):
#             model_ric.train()
#             model_kev.train()
            
#             for idx, data in enumerate(train_loader):
#                 data_ric = copy.deepcopy(data)
#                 data_ric = format_batch_for_ricardo(data_ric)
#                 ret_ric = model_ric.run_on_batch(data_ric, optimizer=optimizer_ric)
#                 wandb.log({
#                     "train/ric_loss": ret_ric['loss'].data.cpu().numpy()
#                 })


#                 data_kev = copy.deepcopy(data)
#                 optimizer_kev.zero_grad()
#                 return_kev = model_kev.training_step(data_kev, idx)
#                 loss_kev = return_kev['loss']
#                 loss_kev.backward()
#                 optimizer_kev.step()

#                 wandb.log({
#                     "train/kev_loss": loss_kev.data.cpu().numpy()
#                 })
                
#             # evaluate models 
#             eval_ric = evaluate(model_ric, val_loader, model_choice="ricardo")
#             wandb.log({
#                 "val/ric_MAE": eval_ric['MAE'],
#                 "val/ric_MRE": eval_ric['MRE'],
#                 "val/ric_AUC": eval_ric['AUC'],
#                 "val/ric_BA": eval_ric['BA'],
#                 "val/ric_F1": eval_ric['F1'],
#                 "val/ric_Recall": eval_ric['Recall'],
#                 "val/ric_loss": eval_ric['loss']
#             })


#             eval_kev = evaluate(model_kev, val_loader, model_choice="kevin")
#             wandb.log({
#                 "val/kev_MAE": eval_kev['MAE'],
#                 "val/kev_MRE": eval_kev['MRE'],
#                 "val/kev_AUC": eval_kev['AUC'],
#                 "val/kev_BA": eval_kev['BA'],
#                 "val/kev_F1": eval_kev['F1'],
#                 "val/kev_Recall": eval_kev['Recall'],
#                 "val/kev_loss": eval_kev['loss']
#             })
                
            
#         wandb.finish()
            
            #true_missing.append(ret['imputations'][0,3,0].data.cpu().numpy().tolist())
    # # summary dataset
    # if rep==0:
    #     # create summary dataset 
    #     df_metrics = pd.DataFrame({'Epoch':range(len(Loss_brits)), 'Loss':Loss_brits,'MAE':MAE_brits,
    #                         'MRE':MRE_brits, 'AUC':AUC_brits, 'BA':BA_brits, 'F1':F1_brits,
    #                         'Recall':Recall_brits, 'Model':model_name})
    #     df_metrics['batch_size'] = batch_size
    #     df_metrics['lr'] = lr
    #     df_metrics['Question'] = question
    #     df_metrics['Facial feature'] = open_face
    #     df_metrics['Ratio missing'] = ratio_missing
    #     df_metrics['Repetition'] = rep
    #     df_metrics['Type missing'] = type_missing
    #     df_metrics['RNN name'] = rnn_name

    # else:
    #     df_metrics1 = pd.DataFrame({'Epoch':range(len(Loss_brits)), 'Loss':Loss_brits,'MAE':MAE_brits,
    #                         'MRE':MRE_brits, 'AUC':AUC_brits, 'BA':BA_brits, 'F1':F1_brits,
    #                         'Recall':Recall_brits, 'SF':SF_brits, 'Model':model_name})
    #     df_metrics1['batch_size'] = batch_size
    #     df_metrics1['lr'] = lr
    #     df_metrics1['Question'] = question
    #     df_metrics1['Facial feature'] = open_face
    #     df_metrics1['Ratio missing'] = ratio_missing
    #     df_metrics1['Repetition'] = rep
    #     df_metrics['Type missing'] = type_missing
    #     df_metrics['RNN name'] = rnn_name

    #     # concatenate
    #     df_metrics = pd.concat([df_metrics, df_metrics1], ignore_index=True)

    # print('End of Experiment!')
    # print('')

    # # save results
    # from pathlib import Path
    # save_path = Path('ricardo_results') / experiment_name / 'df_metrics_'/ f"{question}_{open_face}_bs_{str(batch_size)} _epoch_{str(epochs)}_lr_{str(lr)}_rep_{str(repetitions)}_rm_{str(ratio_missing)}_model_name_{rnn_name}_{type_missing}_v1.csv"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # df_metrics.to_csv(save_path, index=False)
    # # df_metrics.to_csv(f"experiment_name+'/df_metrics_'+question+'_'+open_face+'_bs'+str(batch_size) +'_epoch'+ str(epochs)+'_lr'+str(lr).replace('0.','')+'_rep'+ str(repetitions)+'_rm'+str(ratio_missing).replace('0.','') +'_'+model_name+'_'+rnn_name+'_'+type_missing+'_v1.csv', index=False)
    # print(df_metrics.iloc[:, :-4].tail(5))