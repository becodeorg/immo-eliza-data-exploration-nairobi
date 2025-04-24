import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def rows_columns_count(self):
        counts = [self.df.shape[0], self.df.shape[1]]
        labels = ["Rows", "Columns"]

        plt.style.use('classic')
        fig, ax = plt.subplots()
        bars = ax.bar(labels, counts, color=['g', 'r'], label=labels)
        ax.bar_label(bars, label_type='edge')
        ax.set_xlabel('Categories', fontsize=20)
        ax.set_ylabel('Count', fontsize=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_title('Rows and Columns Count', fontsize=20)
        ax.legend(title='Rows & Columns')

        plt.show()

    def type_distribution(self):
        type_counts = self.df['type'].value_counts()
        labels = type_counts.index
        amounts = type_counts.values
        colors = [(1, 0, 1), (1, 0, 0, 0.8)]
        legend_labels = [f"{label} - {amount}" for label, amount in zip(labels, amounts)]

        plt.style.use('classic')
        fig, ax = plt.subplots()
        wedges, text, autotexts = ax.pie(
            amounts,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops=dict(color='w')
        )
        ax.legend(wedges, legend_labels, title='Amount', prop={'size': 10, 'weight': 'bold'}, loc='best',
                  bbox_to_anchor=(1, 0.5))

        plt.setp(autotexts, size=12, weight="bold")
        ax.set_title('Property Types')
        ax.axis('equal')
        plt.tight_layout()

        plt.show()
