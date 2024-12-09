#from torch.utils.data import Dataset
import os
from typing import Dict, List, Literal, Union
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path, PosixPath
from lightning.pytorch import LightningDataModule
import re
from multitask_missing_q10 import MySet
from multitask_missing_q10 import collate_fn as ricardo_collate_fn

def miss_f(x, n_miss, random = True):
    x1 = x.copy()
    #print('len(x1):', len(x1))   # 40
    if random:
        # index for missing
        m_indx = np.random.choice(list(range(len(x1))), n_miss, replace= False)
    else:
        np.random.seed(14)
        m_indx = np.random.choice(list(range(len(x1))), n_miss, replace= False)

    eval_mask1 = np.zeros(len(x))
    mask1 = np.ones(len(x))
    for i in m_indx:
        x1[i] = 0
        eval_mask1[i] = 1
        mask1[i] = 0

    return x1, eval_mask1.tolist(), mask1

def miss_f2(x, n_miss, random = True):
    # Consecutive Missing Values (CMV)
    x1 = x.copy()
    #print('len(x1):', len(x1))   # 40
    if random:
        # index for missing
        seq_len0 = len(x1)
        m_indx0 = np.random.choice(list(range(seq_len0)), 1, replace= False)[0]
        if (m_indx0 + n_miss) > seq_len0: # superior corner problem
            m_indx0 = m_indx0 - n_miss + 1
        m_indx = list(range(m_indx0,m_indx0+n_miss))
    else:
        np.random.seed(14)
        m_indx = np.random.choice(list(range(len(x1))), n_miss, replace= False)

    eval_mask1 = np.zeros(len(x))
    mask1 = np.ones(len(x))
    for i in m_indx:
        x1[i] = 0
        eval_mask1[i] = 1
        mask1[i] = 0
    return x1, eval_mask1.tolist(), mask1


def mask_f(x):
    x2 = np.zeros((1, len(x)))
    for i, dx in enumerate(x):
        if dx != 0:
            x2[0, i] = 1

    return x2.tolist()[0]

def delta_f(x):
    d1 = np.zeros(len(x))
    m1 = mask_f(x)
    count = 0
    for i in range(len(x)):
        if i > 0 and m1[i] == 1:
            d1[i] = count
        elif i > 0 and m1[i] == 0:
            count += 1
            d1[i] = count
        else:
            d1[i] = 0

    return d1

class DAICWOZDatamodule(LightningDataModule):
    def __init__(
            self, 
            label: Literal['phq8', 'ptsd', 'both'],
            question: str,
            open_face: str,
            delta_steps: int,
            delta_average: bool,
            regen: bool,
            ratio_missing: float,
            type_missing: Literal['Random', 'CMV'],
            batch_size: int,
            num_workers: int,
            ricardo: bool,
            val_ratio: float = 0.0,
            **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = Path(__file__).parent.parent.parent /'data' / 'daicwoz'
        self.data_file_path: PosixPath = self.data_path / "edaic2" / "data.pt"
        self.labels_path: PosixPath = self.data_path / "labels"
        self.open_face_processed_path: PosixPath = self.data_path /  'data_open_face_processed'
        self.label_dict = {'phq8': 0, 'ptsd': 1, 'both': [0, 1]}
        

    @property
    def data_info(self):
        num_features = {
            'landmark': 136,
            'eye_gaze': 12,
            'action_unit': 14,
            'all': 162
        }
        
        return {
            "n_time_steps": 40,
            "n_features": num_features[self.hparams.open_face],
            "n_classes": 2
        }
    
    def prepare_data(self) -> None:
        """
        Prepare all data for to be loaded in the DAICWOZDataset class.

        If the data file exists and the `regen` flag is not set, the method tries to load the data from the file.
        If the file is not found or the `regen` flag is set, the method generates new data and saves it to the data/daicwoz/edaic2/data.pt.

        Returns:
            None
        """
        
        data, labels_phq8, labels_ptsd, index_train, index_test, train_id, test_id = self.generate()
        # outputs = [data, labels]
        outputs = {
            "data": data, 
            "labels_phq8": labels_phq8, 
            "labels_ptsd": labels_ptsd, 
            "index_train": index_train, 
            "index_test": index_test,
            "train_id": train_id,
            "test_id": test_id
        }
        (self.data_file_path.parent).mkdir(parents=True, exist_ok=True)
        torch.save(outputs, self.data_file_path)
        
    
    def generate(self):
        """
        Aggregates the data based on the supplied `question` and `open_face` arguments.

        Returns:
            data (torch.Tensor): The input data tensor of shape (seq_length, batch_size, num_features).
            labels_phq8 (torch.Tensor): The PHQ-8 labels tensor.
            labels_ptsd (torch.Tensor): The PTSD labels tensor.
            index_train (list): The indices of the training data.
            index_test (list): The indices of the testing data.
            train_id (list): The participant IDs of the training data.
            test_id (list): The participant IDs of the testing data.
        """
        
        train_label: pd.DataFrame = pd.read_csv(self.labels_path / self.hparams.question / 'train5.csv')
        test_label: pd.DataFrame = pd.read_csv(self.labels_path / self.hparams.question / 'test5.csv')

        # train sessions
        train_sessions: List = [str(i) for i in train_label.Participant_ID.tolist()]

        # test sessions
        test_sessions: List = [str(i) for i in test_label.Participant_ID.tolist()]

        sessions: List = train_sessions + test_sessions
        xt_tensor: List = []
        bad_sessions: List = []
        count = 0
        seq_length: int = int(40/self.hparams.delta_steps) #TODO this is assuming 40 is the sequence length, get the value from data / can keep, since times steps can be known before hand
        
        # to scale variables
        from sklearn.preprocessing import FunctionTransformer
        scaler = FunctionTransformer(lambda x: x) 
        # scaler = MinMaxScaler()
        for session in sessions:
            #print(session)
            #id_list.append(session)
            # get face features 
            try:
                if self.hparams.open_face=='all':
                    # landmark
                    df_land: pd.DataFrame = pd.read_csv(self.open_face_processed_path / 'landmark' / self.hparams.question / f"{session}.csv")
                    df_land1: pd.DataFrame = df_land.iloc[:,0:1]
                    
                    try:
                        df_land2: pd.DataFrame = pd.DataFrame(scaler.fit_transform(df_land.iloc[:,1:]))   # scale inputs
                    except:
                        df_land2: pd.DataFrame = df_land.iloc[:,1:]
                    df_land: pd.DataFrame = pd.concat([df_land1,df_land2], axis=1)
                    
                    # eye gaze
                    df_gaze: pd.DataFrame = pd.read_csv(self.open_face_processed_path / 'eye_gaze' / self.hparams.question / f"{session}.csv")
                    try:
                        df_gaze: pd.DataFrame = pd.DataFrame(scaler.fit_transform(df_gaze.iloc[:,1:]))   # scale inputs
                    except:
                        df_gaze: pd.DataFrame = df_gaze.iloc[:,1:]
                    
                    # action unit
                    df_au: pd.DataFrame = pd.read_csv(self.open_face_processed_path / 'action_unit' / self.hparams.question / f"{session}.csv")
                    try:
                        df_au: pd.DataFrame = pd.DataFrame(scaler.fit_transform(df_au.iloc[:,1:]))   # scale inputs
                    except:
                        df_au: pd.DataFrame = df_au.iloc[:,1:]

                    #df1 = pd.concat([df_land, df_gaze.iloc[:,1:],df_au.iloc[:,1:]], axis=1)
                    df1: pd.DataFrame = pd.concat([df_land, df_gaze, df_au], axis=1)
                else:
                    df1: pd.DataFrame = pd.read_csv(self.open_face_processed_path / self.hparams.open_face / self.hparams.question / f"{session}.csv")
                    length_series: int = len(df1)
                    if self.hparams.delta_average:
                        # create id columns with sequences
                        result1 = []
                        for d in range(int(length_series/self.hparams.delta_steps+1)):
                            result1.extend(self.hparams.delta_steps * [d])
                        df1['colsum1'] = result1[0:length_series]
                        df1: pd.DataFrame = df1.groupby(['colsum1']).mean()
                    else:
                        list_sample_time_steps: List = list(range(0, length_series,self.hparams.delta_steps))
                        df1: pd.DataFrame = df1.iloc[list_sample_time_steps]
                # drop rows with zero: error o no information captured
                sum_rows1: List = df1.iloc[:,1:].sum(axis=1).tolist()
                indices: List = [i for i, x in enumerate(sum_rows1) if x ==0]
                df1: pd.DataFrame = df1.drop(df1.index[indices])

                # remove timestamp
                df1: pd.DataFrame = df1.iloc[:,1:]
                if len(df1)>=seq_length:
                    # first i-th 
                    df1 = df1.iloc[0:seq_length,:]
                elif len(df1)<seq_length:
                    #print('len(df1).shape:',len(df1))
                    #print('seq_length:',seq_length)
                    # add zeros (pandas or tensor)
                    # creat matrix with zeros 
                    matrix_zeros: np.ndarray = np.zeros((seq_length -len(df1), df1.shape[1]))
                    df_zeros: pd.DataFrame = pd.DataFrame(matrix_zeros, columns=df1.columns)
                    df1: pd.DataFrame = pd.concat([df1, df_zeros],ignore_index=True)
                    #df1 = df1.append(df_zeros, ignore_index=True)
                # add dataset to list 
                xt_tensor.append(df1.to_numpy())

            except:
                print('Problem with id:', session)
                bad_sessions.append(session)
                count += 1
                #time_steps_list.append(0)
                #seconds_list.append(0) 

        # labels PHQ8
        #train label is 5 column dataframe, has id as first, phq score, ptsd severity, label (0,1) for phq, label(0,1) for ptsd
        labels1: List[int] = train_label.label_phq8_10.values.tolist() 
        #labels 1 is the series of phq labels for train data
        
        labels2: List = test_label.label_phq8_10.values.tolist()
        #labels 2 is the series of phq labels for test data
        
        # label PTSD
        label_ptsd1: List = train_label.label_ptsd_36.values.tolist()
        label_ptsd2: List = test_label.label_ptsd_36.values.tolist()
        
        # remove some bad sessions lables 
        labels_phq8: List = labels1 + labels2 #combine all phq scores
        labels_ptsd: List = label_ptsd1 + label_ptsd2 #combine all ptsd scores
        
        # torch tensor
        labels_phq8: torch.Tensor = torch.tensor(labels_phq8, dtype=torch.float)
        labels_ptsd: torch.Tensor = torch.tensor(labels_ptsd, dtype=torch.float)
        
        #print('labels.shape:',labels.shape)
        index_train: List = list(range(0, len(labels1))) #all train are the first batch of the combined labels_phq8 from above
        index_test: List = list(range(len(labels1), len(labels1) + len(labels2))) #all test are the second batch of the combined labels_phq8 from above
        xt_tensor = np.array(xt_tensor)
        xt_tensor: torch.Tensor = torch.from_numpy(xt_tensor).float()
        data: torch.Tensor = xt_tensor.permute(0, 2, 1)
        train_id: List = train_label.Participant_ID.tolist()  
        test_id: List = test_label.Participant_ID.tolist()      
        #print('data.shape:',data.shape) #[seq, batch, features]
        
        return data, labels_phq8, labels_ptsd, index_train, index_test, train_id, test_id #get train_id, test_id, which is id_train = df_label[df_label['is_train']==1].id.tolist() from daicwoz.py
    
    def setup(self, stage=None):
        """
        Set up the data module. This method creates the final Dataset obect to be split into 
        training and testing sets.

        Args:
            stage: Optional[str]: The stage of the data module (e.g., 'fit', 'test', 'predict').

        Returns:
            None
        """
        self.dataset = DAICWOZDataset(
            target=self.hparams.label,
            ratio_missing=self.hparams.ratio_missing,
            type_missing=self.hparams.type_missing
        )
        self.train_dataset = [sample for sample in self.dataset if sample['is_train'] == 1]
        
        if self.hparams.ricardo:
            self.val_dataset = [sample for sample in self.dataset if sample['is_train'] == 0]
            self.test_dataset = torch.tensor([0.0])
        elif not self.hparams.ricardo and self.hparams.val_ratio == 0.0:
            self.val_dataset = torch.tensor([0.0])
            self.test_dataset = [sample for sample in self.dataset if sample['is_train'] == 0]
        else:
            import random
            self.train_dataset = random.sample(self.train_dataset, int(len(self.train_dataset) * (1 - self.hparams.val_ratio)))
            self.val_dataset = random.sample(self.train_dataset, int(len(self.train_dataset) * self.hparams.val_ratio))
            self.test_dataset = [sample for sample in self.dataset if sample['is_train'] == 0]

            
    def train_dataloader(self): #-> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False)
    

        
class DAICWOZDataset(Dataset):
    
    def __init__(
            self, 
            target: Union[str, List[str]] = 'both',
            ratio_missing: float = 0.1,
            type_missing: Literal['Random', 'CMV'] = "Random"
            ):
        super().__init__()


        self.data_path = Path(__file__).parent.parent.parent / 'data' / 'daicwoz'
        self.data_file_path = self.data_path / "edaic2" / "data.pt"
        self.all_data = torch.load(self.data_file_path)
        assert self.data_file_path.exists(), "Data file not found. Please run the datamodule.prepare_data() to generate the data."
        # check that the keys 'data', 'labels_phq8', 'labels_ptsd', 'index_train', 'index_test' are in the file
        assert all([key in self.all_data.keys() for key in ['data', 'labels_phq8', 'labels_ptsd', 'index_train', 'index_test', 'train_id', 'test_id']]), "Data file does not contain the required keys."
        self.train_indices = self.all_data['index_train']
        self.test_indices = self.all_data['index_test']
        self.train_id = self.all_data['train_id']
        self.test_id = self.all_data['test_id']
        self.target = target
        self.type_missing = type_missing
        
        self.label_dict = {'phq8': 0, 'ptsd': 1, 'both': [0, 1]}
        self.data_with_missing = self.input_missing(ratio_missing, type_missing)

    def __len__(self):
        return len(self.data_with_missing['X'])
    
    def __getitem__(self, idx):
        """Extract an example from a dataset.
        
        Parameters
        ----------
        idx : int
            Integer indexing the location of an example.

        Returns
        -------
        rec : dict
            A dictionary containing the data, forward and backward data with missing values, and the labels.
        """
        rec = self.data_with_missing[idx]
        if idx in self.test_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec
    
    def input_missing(self, ratio_missing: float, type_missing: Literal['Random', 'CMV']):
        data0 = []
        n_miss = int(self.all_data['data'].shape[-1] * ratio_missing) # this is assuming time steps is last, should this always hold?
        for i in range(len(self.all_data['data'])):
            data0.append(
                self.convert(
                    self.all_data['data'].detach().numpy()[i],
                    [
                        self.all_data['labels_phq8'][i], 
                        self.all_data['labels_ptsd'][i]
                    ], 
                    n_miss, 
                    type_missing))
        return data0

    def convert(
            self,
            x: torch.Tensor, 
            y: List[torch.Tensor],
            n_miss: float, 
            type_missing: Literal['Random', 'CMV'] ='Random'
        ):
        # input : x[n_features[n_tiempo]]
        n_feature = len(x) 
        #print('n_feature:',n_feature)  # 12
        n_time = len(x[0])            
        #print('n_time:', n_time)       # 40
        
        # forwards: initial values
        x0 = np.zeros((n_feature, n_time))
        values_f = x0.copy()
        masks_f = x0.copy()
        deltas_f = x0.copy()
        
        evals_f = x0.copy() 
        eval_masks_f = x0.copy()
        
        # backward: initial values
        values_b = x0.copy() 
        masks_b = x0.copy()
        deltas_b = x0.copy()
        
        evals_b = x0.copy() 
        eval_masks_b = x0.copy()
        
        for i in range(n_feature):
            if type_missing=='Random':
                x_miss = miss_f(x[i], n_miss, True)
            elif type_missing=='CMV': # Consecutive Missing Values (CMV)
                x_miss = miss_f2(x[i], n_miss, True) 
            # Forwards
            values_f[i,:] = x_miss[0]
            #masks_f[i,:] = mask_f(x_miss[0])
            masks_f[i,:] = x_miss[2]
            deltas_f[i,:] = delta_f(x_miss[0])
            
            evals_f[i,:] = x[i] 
            eval_masks_f[i,:] = x_miss[1]
            
            # Backward
            values_b[i,:] = np.flip(x_miss[0], axis = 0).tolist() 
            masks_b[i,:] = np.flip(mask_f(x_miss[0]), axis = 0).tolist()
            deltas_b[i,:] = delta_f(x_miss[0]) * -1
            
            evals_b[i,:] = np.flip(x[i], axis = 0).tolist() 
            eval_masks_b[i,:] = np.flip(x_miss[1], axis = 0).tolist()
        # Create format to data loader
        X = values_f.T
        missing_mask = masks_f.T
        deltas = deltas_f.T
        X_ori = evals_f.T
        indicating_mask = eval_masks_f.T
        back_X = values_b.T
        back_missing_mask = masks_b.T
        back_deltas = deltas_b.T
        y = torch.tensor(y)[self.label_dict[self.target]]
        data = {
            "X": X, 
            "missing_mask": missing_mask, 
            "deltas": deltas, 
            "X_ori": X_ori, 
            "indicating_mask": indicating_mask, 
            "back_X": back_X, 
            "back_missing_mask": back_missing_mask, 
            "back_deltas": back_deltas, 
            "label": y}
        return data
        


# if __name__ == "__main__":
#     from lightning.pytorch import seed_everything
#     seed_everything(42)

#     dm = DAICWOZDatamodule(
#         label='both',
#         IS_BRITS=True,
#         question='easy_sleep',
#         open_face='eye_gaze',
#         delta_steps=1,
#         delta_average=False,
#         ratio_missing=0.1,
#         type_missing='Random',
#         regen=True,
#         batch_size=8,
#         num_workers=0
#     )
#     dm.prepare_data()
#     dm.setup()
#     loader = dm.train_dataloader()
#     batch: Dict = next(iter(loader))
#     print(batch['forward'])