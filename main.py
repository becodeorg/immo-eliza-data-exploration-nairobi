from utils.correlation import Correlation
from utils.heatmap import DrawHeatmap
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv('datasets/cleaned_kangaroo.csv')

    # draw heatmap to numerical data for observation
    hm = DrawHeatmap(df)
    hm.heatmap(by = 'whole')
    hm.heatmap(by = 'type')
    hm.heatmap(by = 'region')

    # pre-processing data before corr and select features
    df_proc = Correlation(df)

    df_proc.add_col_region() # region may matter as observed from heatmap
    df_proc.one_hot_encoding()
    df_proc.target_encoding()
    df_proc.label_encoding()
    df_proc.fillnato0()

    df_proc.corr_cals()
    df_proc.select_features(thres1=0.1, thres2=0.3)