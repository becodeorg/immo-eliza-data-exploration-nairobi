import pandas as pd
from utils.data_processing import DataCleaner
from utils.visualization import DataVisualizer

if __name__ == '__main__':
    file_path = 'datasets/Kangaroo.csv'
    write_path = 'datasets/cleaned_kangaroo.csv'
    cleaner = DataCleaner(file_path)
    cleaner.remove_columns_by_missing_percentage(100, to_drop='url')
    cleaner.remove_duplicates('id')
    cleaner.remove_spaces()
    cleaner.remove_by_column_values('type', ['Apartment_group', 'House_group'])
    cleaner.handle_errors()
    cleaner.handle_missing_values()
    #cleaner.write_to_csv(write_path)
    clean_data = cleaner.df

    visualizer = DataVisualizer(clean_data)
    visualizer.plot_property_types_by_province()
    visualizer.most_least_expensive_locality('Flanders')
    visualizer.most_least_expensive_locality('Wallonia', metric='price_per_m2', top=12)
    visualizer.most_least_expensive_locality('Brussels', metric='median_price', type_of_sort=True)


