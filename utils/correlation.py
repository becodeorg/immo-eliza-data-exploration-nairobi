import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

class Correlation:
    '''A class to perform correlation analysis to a pd.DataFrame, including pre-processing(e.g., encoding), corr calculation and feature selection.

    Attributes:
        None.

    Methods:
        add_col_region() -> None
        one_hot_encoding(exclude_cols: list[str] =['locality','epcScore','floodZoneType','buildingCondition']) -> None
        target_encoding(target_cols: list[str] = ['postCode']) -> None
        label_encoding(label_cols: list[str] = ['floodZoneType','buildingCondition','epcScore']) -> None
        fillnato0() -> None
        corr_cals() -> None
        select_features(thres1: float, thres2: float) -> None
    '''

    def __init__(self, df: pd.DataFrame) -> None:
        '''Initializes a Correlation instance with a given pd.DataFrame.

        Args:
            df (pd.DataFrame): the dataframe to be analyzed.

        Returns:
            None.
        '''
        self.df = df

    def add_col_region(self) -> None:
        '''Add a column region to the dataframe.

        Args:
            None.

        Returns:
            None.
        '''
        self.df['region'] = self.df['province'].map(lambda x: 
                                                    'Flanders' if x in ['West flanders', 'East flanders', 'Antwerp', 'Flemish brabant'] else (
                                                        'Walloon' if x in ['Hainaut', 'Liège', 'Walloon brabant', 'Namur', 'Luxembourg'] else 
                                                        'Brussels'))

    def one_hot_encoding(self, exclude_cols: list[str] =['locality','epcScore','floodZoneType','buildingCondition']) -> None: 
        '''Apply one-hot encoding (0/1) to all object-type columns inplace.

        Args:
            exclude_cols (list[str]): object-type columns excluded from one-hot encoding.
        
        Returns:
            None.
        '''
        encoding_cols = list(self.df.select_dtypes(include=['object']).columns)
        for col in exclude_cols:
            encoding_cols.remove(col)
        self.df[encoding_cols] = self.df[encoding_cols].fillna('unknown')
        print(f'end of one_hot_encoding these columns: {encoding_cols}')

    def target_encoding(self, target_cols: list[str] = ['postCode']) -> None:
        '''Apply target encoding (mean value) to high-cardinality columns inplace.

        Args:
            target_cols (list[str]): columns to be target encoding. Defaults to 'postCode'.
        
        Returns:
            None.
        '''
        # attention: use 'postCode' instead of 'locality', more precise to present location info!
        for col in target_cols:
            self.df[col + '_target_encoding'] = self.df[col].map(self.df.groupby(col)['price'].mean())
        print(f'end of target_encoding these columns: {target_cols}')

            
    def label_encoding(self, label_cols: list[str] = ['floodZoneType','buildingCondition','epcScore']) -> None:
        '''Apply label encoding (0,1,2...) to columns with rankable columns inplace.

        Args:
            label_cols (list[str]): columns to be label encoding. Defaults to ['floodZoneType', 'buildingCondition', 'epcScore'].
        
        Returns:
            None.
        '''
        # for defalut cols, label order is pre-defined with prior knowledge
        if label_cols == ['floodZoneType','building_ranking','epcScore']:
            order_flood = {'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 0,
                           'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 1,
                            'RECOGNIZED_FLOOD_ZONE': 2,
                            'CIRCUMSCRIBED_WATERSIDE_ZONE': 3,
                            'CIRCUMSCRIBED_FLOOD_ZONE': 4,
                            'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 5,
                            'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 6,
                            'POSSIBLE_FLOOD_ZONE': 7,
                            'NON_FLOOD_ZONE': 8}
            order_building = {'TO_RESTORE': 0,
                              'TO_RENOVATE': 1,
                              'TO_BE_DONE_UP': 2,
                              'GOOD': 3,
                              'JUST_RENOVATED': 4,
                              'AS_NEW': 5}
            order_epcScore = {'G': 0,
                              'F': 1,
                              'E': 2,
                              'D': 3,
                              'C': 4,
                              'B': 5,
                              'A': 6,
                              'A+': 7,
                              'A++': 8,
                              'G_C': None,'G_E': None,'E_C': None,'D_C': None,'C_B': None,'F_D': None,'F_C': None,'C_A': None,'E_D': None,'G_F': None
                              } # attention: epcScores vary by region — the same score may represent different emission levels and thus affect prices differently.
                                # however, since epcScores themselves (not their emission values) are also used for tax, rental indexing, etc., we prefer to treat them uniformly across regions. 
            self.df['floodZoneType_label_encoding'] = self.df['floodZoneType'].map(order_flood)
            self.df['buildingCondition_label_encoding'] = self.df['buildingCondition'].map(order_building)
            self.df['epcScore_label_encoding'] = self.df['epcScore'].map(order_epcScore)
        
        else: # for customized cols, sklearn label cols by the order of their appearance.
            encoder = sklearn.preprocessing.LabelEncoder()
            for col in label_cols:
                self.df[col] = encoder.fit_transform(self.df[col])
        print(f'end of label_encoding these columns: {label_cols}')


    def fillnato0(self) -> None:
        '''Fill missing values(np.NaN) in hasXXX columns(e.g., hasBasement) in place.

        Args:
            None.
        
        Returns:
            None.
        '''
        hasXXX_cols = [name for name in list(self.df.columns) if name.startswith('has')]
        self.df[hasXXX_cols] = self.df[hasXXX_cols].fillna(0)
        print(f'end of fillnato0 these columns: {hasXXX_cols}')


    def corr_cals(self) -> None:
        '''Create pearson correlation dataframe for the numerical cols (class int ✔, class float ✔, with np.NaN ✔) as instance attribute. 'Price' is ordered as the first column.

        Args:
            None.
        
        Returns:
            None.
        '''
        df = self.df.select_dtypes(include=['float','int'])
        reorder_cols = ['price'] + [col for col in df.columns if col != 'price']
        df = df.loc[:, reorder_cols]
        corr = df.corr()
        print('end of corr_cals()')
        self.corr = corr


    def select_features(self, thres1: float, thres2: float) -> dict:
        '''Display a list of variables that are strongly correlated to price and not strongly inter-correlated.

        Args:
            thres1 (float): threshold for price-variable correlation as 'strongly correlated'.
            thres2 (float): threshold for variable-variable correlation as 'strongly correlated'.
        
        Returns:
            None.
        '''
        # select vars which are strongly correlated with 'price' (>= thres1)
        select = self.corr['price'][abs(self.corr['price']) >= thres1].to_dict()
        select.pop('price')
        select = dict(sorted(select.items(), key = lambda item: abs(item[1]), reverse=True)) # reorder selected vars by their correlation values
        # check if these vars are strongly inter-correlated (>= thres2), if so, drop the one less correlated to 'price'
        select_final = select.copy()
        for var1, val1 in select.items():
            for var2, val2 in select.items():
                if var1 != var2:
                    if self.corr.loc[var1, var2] >= thres2:
                        if abs(val1) > abs(val2):
                            select_final.pop(var2, None)
                        else:
                            select_final.pop(var1, None)
        
        print('end of select_features(), the following are selected:')
        for var, val in select_final.items():
            print(f'{var}: {val:.2f}')