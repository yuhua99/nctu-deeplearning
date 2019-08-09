import showstuff
import os
import pandas as pd

class AccuracyResult():
    def __init__(self, fileName = "AccuracyResult.csv"):
        self.filepath = os.path.join('.', fileName)
        # print(self.filepath)
        if os.path.isfile(self.filepath):
            self.df = pd.read_csv(self.filepath, index_col=0)
        else:
            self.df = pd.DataFrame(columns = ["ReLU", "Leaky ReLU", "ELU"])
        self.mapping = {
            'ELU' : 'elu',
            'ReLU' : 'relu',
            'Leaky ReLU' : 'leaky_relu',
        }
        
    def add(self, modelName, Accs, show=False):
        rows = [0.0]*len(self.df.columns)
        if modelName in self.df.index:
            rows = self.df.loc[modelName]
        for idx, col in enumerate(self.df.columns):
            if Accs[self.mapping[col] + '_test']:
                acc = max(Accs[self.mapping[col] + '_test'])
                if acc > rows[idx]:
                    rows[idx] = acc 
                
        if len(rows) != len(self.df.columns):
            raise AttributeError("Not enougth columns")
        
        self.df.loc[modelName] = rows
        
        #self.df.to_csv(self.filepath)
        
        if show:
            fig = showstuff.showAccuracy(
                title=modelName,
                **Accs
            )
            fig.savefig(
                fname=os.path.join('.', modelName + '.png'),
                dpi=300,
                metadata = {
                    'Title': modelName,
                    'Author': '0756110',
                },
                bbox_inches="tight",
            )
        
    def show(self):
        print(self.df)
    
    def clear(self):
        self.df = self.df.iloc[0:0]
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)