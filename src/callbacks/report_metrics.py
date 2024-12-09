from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing import Literal
import os
import glob
import pandas as pd
from pathlib import Path
from lightning.pytorch.utilities import rank_zero_only


class Table3Callback(Callback):
    def __init__(self, experiment_path):
        super().__init__()
        self.experiment_path = Path(experiment_path).parents[2]

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        new_df = self.produce_results_table(self.experiment_path)
        pl_module.print(new_df)
        new_df.to_csv(self.experiment_path / 'results_table.csv')
        new_df.to_latex(self.experiment_path / 'results_table.tex')
    

    def produce_results_table(self, experiment_path):
        # find all instances of a csv file in all possible subdirectories
        path = experiment_path
        csv_files = list(path.rglob("*metrics.csv"))

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            df['question'] = file.parents[4].name
            df['ratio_missing'] = file.parents[3].name
            df['seed'] = file.parents[2].name
            dfs.append(df)

        df = pd.concat(dfs)

        pivot = df.pivot_table(index=['question', 'ratio_missing', 'seed'], values='val/f1', aggfunc=['max', 'std'])
        # take the mean of the question/ratio_missing combination 
        pivot = pivot.groupby(['question', 'ratio_missing']).mean()
        # pivot.to_csv(path / 'results_table1.csv')

        pivot2 = pivot.unstack(level=1)
        pivot2['max_score_other'] = pivot2['max']['val/f1'].drop(columns=['0.0']).max(axis=1)

        # Calculate the gain
        pivot2['gain'] = pivot2['max_score_other'] - pivot2['max']['val/f1']['0.0']
        # remove the max column
        pivot2 = pivot2.drop(columns='max_score_other')

        # make every value a string
        pivot2 = pivot2.map(lambda x: f'{x:.2f}')
        # make every value that falls under 'max' column equal to 1
        new_df = pivot2.loc[:, 'max'] + " ± " + pivot2.loc[:, 'std'] 

        new_df['gain'] = pivot2['gain']
        new_df.loc['Avg'] = 0
        new_df.loc['Max'] = 0 

        # import pdb; pdb.set_trace() #TODO getting a nan in one of these iterations
        new_df2 = new_df.iloc[:-2].map(lambda x: x.split("±")[0]).apply(pd.to_numeric)
        new_df2.loc['Avg.'] = new_df2.mean()
        new_df2.loc['Max'] = new_df2.max()

        new_df.loc['Avg'] = new_df2.loc['Avg.']
        new_df.loc['Max'] = new_df2.loc['Max']
        return new_df



        # pivot.to_csv(path / 'results_table.csv')
        # print(pivot)
# class ReportMetrics(Callback):
#     def __init__(self, hydra_path, metric: Literal["f1", "auc"]):
#         self.hydra_path = hydra_path
#         self.metric = metric
    
#     def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         path = "/home/khickey/test_impute/logs/debug/05-02-16-15-57"
#         # check if a .csv file exists in that directory
#         final_results_csv = glob.glob(path + '/*.csv')
#         if final_results_csv:
#             # read the csv file
#             with open(final_results_csv[0], 'r') as f:
#                 lines = f.readlines()
#             # get the last line
#             last_line = lines[-1]
#             # get the metric value
#             metric_value = last_line.split(',')[1]
#             # log the metric value
#             trainer.logger.log_metrics({self.metric: float(metric_value)})
#         else:
#             print("No .csv file found in the directory")