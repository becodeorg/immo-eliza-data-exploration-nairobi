import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


class DataVisualizer:
    """
    A class to visualize real estate data
    """
    def __init__(self, df):
        """
        Initializes the DataVisualizer object with the DataFrame and regions for Flanders, Wallonia, and Brussels.
        :param df: pandas DataFrame, the original dataset containing real estate data
        """
        self.df = df
        self.flanders_provinces = ['West flanders', 'East flanders', 'Antwerp', 'Flemish brabant']
        self.wallonia_provinces = ['Hainaut', 'Liège', 'Walloon brabant', 'Namur', 'Luxembourg']
        self.brussels_province = 'Brussels'

    def filter_by_region(self, region: str) -> pd.DataFrame:
        """
        Filters the data by region (Flanders, Wallonia, or Brussels).
        :param region: str, the name of the region ('Flanders', 'Wallonia', 'Brussels')
        :return: pandas DataFrame, the filtered DataFrame based on the region
        :raises ValueError: if an invalid region is provided
        """
        if region == 'Flanders':
            return self.df[self.df['province'].isin(self.flanders_provinces)]
        elif region == 'Wallonia':
            return self.df[self.df['province'].isin(self.wallonia_provinces)]
        elif region == 'Brussels':
            return self.df[self.df['province'] == self.brussels_province]
        else:
            raise ValueError("Region must be 'Flanders', 'Wallonia', or 'Brussels'")

    @staticmethod
    def format_large_numbers(x: int | float, pos) -> str:
        """
        Formats numbers with thousands separators for better readability.
        :param x: number to be formatted
        :return: str, formatted number as a string
        """
        return f'{x:,.0f}'

    def collect_data(self, region: str) -> pd.DataFrame:
        """
        Collects data on localities for the specified region.
        :param region: str, the name of the region ('Flanders', 'Wallonia', 'Brussels')
        :return: pandas DataFrame, collected data by locality
        """
        filtered_data = self.filter_by_region(region)
        filtered_data['price_per_m2'] = filtered_data['price'] / filtered_data['habitableSurface']
        collected_data = filtered_data.groupby('locality').agg(
            average_price=('price', 'mean'),
            median_price=('price', 'median'),
            price_per_m2=('price_per_m2', 'mean')
        ).reset_index()
        return collected_data

    def most_least_expensive_locality(self, region: str, metric: str = 'average_price', type_of_sort: bool = False,
                                      top: int = 10) -> None:
        """
        Plots a graph for the most expensive or least expensive localities in the given region.
        :param region: str, the name of the region ('Flanders', 'Wallonia', 'Brussels')
        :param metric: str, the metric to sort by ('average_price', 'median_price', 'price_per_m2') default is 'average_price'
        :param type_of_sort: bool, if True, sort in ascending order; if False, sort in descending order
        :param top: int, the number of top localities to display (default is 10)
        :return: None
        """
        if metric not in ('average_price', 'median_price', 'price_per_m2'):
            raise ValueError("Metric must be one of: 'average_price', 'median_price', 'price_per_m2'")

        collect_data = self.collect_data(region)
        most_expensive = collect_data.sort_values('average_price', ascending=type_of_sort).head(top)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=metric, y='locality', data=most_expensive, palette='Blues_r' if type_of_sort else 'Blues_d')
        order = 'Least' if type_of_sort else 'Most'
        plt.title(f"{order} Expensive Locality in {region} ({metric.replace('_', ' ').title()})")
        plt.xlabel(f"{metric.replace('_', ' ').title()} (€)")
        plt.ylabel('Locality')
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_large_numbers))

        #  Automatically adjusts the arrangement of elements on the chart
        plt.tight_layout()
        plt.show()

    def plot_property_types_by_province(self) -> None:
        """
        Plots a graph showing the number of houses and apartments in each province.
        :return: None
        """

        # Grouping data by province and property type
        property_amount = self.df.groupby(['province', 'type']).size().unstack(fill_value=0)

        # Reset index
        property_amount = property_amount.reset_index()

        # melt turns a "wide" table into a "long" one. We transform each column
        # (property type) into a separate row with a type label and a value.
        property_amounts_melted = property_amount.melt(id_vars="province", value_vars=property_amount.columns[1:],
                                                       var_name='Property Type', value_name='Count')

        plt.figure(figsize=(12, 8))
        sns.barplot(x='province', y='Count', hue='Property Type', data=property_amounts_melted,
                    palette=self.choose_random_palette_theme())

        plt.title('Amount of Houses and Apartments by Province', fontsize=18, fontweight='bold', fontstyle='italic',
                  family='fantasy')
        plt.xlabel('Province', fontsize=14, fontweight='bold', fontstyle='oblique', family='monospace')
        plt.ylabel('Amount of Properties', fontsize=14, fontweight='bold', fontstyle='oblique', family='monospace')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Property Type', loc='upper left')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def choose_random_palette_theme() -> str:
        """
        Chooses a random categorical palette theme from Seaborn's predefined palettes.
        :return: str, a random palette theme name
        """
        categorical_palettes = list(sns.palettes.SEABORN_PALETTES.keys())

        return random.choice(categorical_palettes)
