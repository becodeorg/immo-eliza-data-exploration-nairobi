import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DrawHeatmap:
    def __init__(self, df: pd.DataFrame) -> None:
        '''Initialize a DrawHeatmap instance with a given pd.DataFrame.

        Args:
            df (pd.DataFrame): the dataframe to be visualized.
        
        Returns:
            None
        '''
        self.df = df
        self.flanders_provinces = ['West flanders', 'East flanders', 'Antwerp', 'Flemish brabant']
        self.wallonia_provinces = ['Hainaut', 'Liège', 'Walloon brabant', 'Namur', 'Luxembourg']
        self.brussels_province = 'Brussels'

    def heatmap(self, by: str ='whole', figsize: tuple = (8, 6), fontsize: int = 5, xtickrotation: int = 15, hspace : float = 0.4) -> None:
        '''Draw heatmap, can draw one plot for all data or subplots by category.

        Args:
            by (str): the col whose unique values used as category, e.g. 'type' -> category: apartment/house. Defaults to 'whole' for all data.
            figsize (tuple): customize figure size. Defaults to (8, 6).
            fontsize (int): customize fontsize for annots and xticklabels. Defaults to 5.
            xtickrotation (int): customize xtickrotation. Defaults to 15.
            hspace (float): customize horizontal space between plots. Defaults to 0.4.
        Returns:
            None.
        '''
        # add a col 'region'
        self.df['region'] = self.df['province'].map(lambda x: 'Flanders' if x in self.flanders_provinces else ('Walloon' if x in self.wallonia_provinces else 'Brussels'))
        # drop one-value col to avoid blank in heatmap
        df = self.df.loc[:, self.df.apply(lambda col: col.nunique(dropna=True) > 1)]
        # reorder 'price' to the first column for clearer heatmap display
        reorder_cols = ['price'] + [col for col in df.columns if col != 'price']
        df = df.loc[:, reorder_cols]

        # prepare corr matrixes
        if by == 'whole':
            val_list = ['- for all data']
            df_by_val = [df]
        else:
            val_list = [val for val in df[by].fillna('Unknown').unique()]
            df_by_val = [df[df[by] == val] for val in val_list]
        corr_list = [df.select_dtypes(include=['float','int']).corr() for df in df_by_val] # corr only applies to numeric cols

        # start drawing heatmap with subplots
        axes_num = len(corr_list)
        fig, axes = plt.subplots(axes_num, 1, figsize=figsize)
        axes = np.atleast_1d(axes) # when axes_num = 1, make sure axes is a np.ndarray for looping 

        # draw super title
        if by is None:
            fig.suptitle('Correlation Heatmap', fontsize=15)
        else:
            fig.suptitle(f'Correlation Heatmap by {by}', fontsize=15)

        # loop to draw subplots
        for val, corr, ax in zip(val_list, corr_list, axes):
            if val == '- for all data':
                ax.set_title(f'{val}', loc='right')
            else:
                ax.set_title(f'{val} heatmap', loc='center')
            mask = np.triu(np.ones_like(corr))
            sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f', annot_kws={'size': fontsize}, mask=mask)
            ax.tick_params('both', labelsize=fontsize)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=xtickrotation)

        plt.subplots_adjust(hspace=hspace)
        plt.show()