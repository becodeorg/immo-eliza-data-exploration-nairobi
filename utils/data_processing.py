import pandas as pd


class DataCleaner:
    """
    A class for loading, cleaning, and saving tabular data from CSV files.
    """

    def __init__(self, filepath: str, sep=',', encoding='utf-8'):
        """
        Initializes the DataCleaner instance by loading data from the specified CSV file.
        :param filepath: The path to the CSV file.
        :param sep: The delimiter used in the file (default is ',').
        :param encoding: The encoding of the file (default is 'utf-8').
        """
        self.original_df = self.create_dataframe(filepath, sep=sep, encoding=encoding)
        self.df = self.original_df.copy()

    @staticmethod
    def create_dataframe(filepath: str, sep: str, encoding: str) -> pd.DataFrame:
        """
        Loads a CSV file into a DataFrame with error handling.
        :param filepath: The path to the file.
        :param sep: The delimiter used in the file.
        :param encoding: The encoding of the file.
        :return: pd.DataFrame: pandas DataFrame with loaded data from csv file.
        """
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=encoding, index_col=0)
            print(f"File {filepath} uploaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at path: {filepath}")
        except pd.errors.ParserError:
            raise pd.errors.ParserError(f"Unable to parse CSV file: {filepath}")
        except UnicodeError:
            raise UnicodeError(f"Encoding error reading file: {filepath}")
        except Exception as e:
            raise Exception(f"Unknown error loading file {filepath}: {str(e)}")

        return df

    def remove_columns_by_missing_percentage(self, percent: int = 60, exceptions: list[str] = None,
                                             to_drop=None) -> pd.Index:
        """
        Removes columns with missing values exceeding the specified percentage.
        Optionally excludes certain columns from being dropped.


        :param percent: The percentage threshold of missing values. Columns with missing values greater than or equal to this percentage will be dropped (default is 60%).
        :param exceptions: A list of column names that should not be dropped, even if they have a high percentage of missing values.
        :param to_drop: column you want to drop at any case (default is None)
        :return: pd.Index: The percentage of missing values for each column.
        """
        if exceptions is None:
            exceptions = []
        missing_percentage = ((self.df.isnull().sum() / len(self.df)) * 100)
        columns_to_drop = [
            col for col, perc in missing_percentage.items()
            if perc >= percent and col not in exceptions
        ]
        self.df = self.df.drop(columns=columns_to_drop)
        if to_drop:
            self.df = self.df.drop(columns=to_drop)

        return missing_percentage

    def remove_duplicates(self, subset=None, keep='first') -> None:
        """
        Removes duplicate rows from the DataFrame.
        :param subset: Name of columns to check for duplicates. If None, checks all columns.
        :param keep: Which duplicate to keep: 'first', 'last', or False: removes all duplicates (default is 'first').
        :return: None
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)

    def remove_spaces(self) -> None:
        """
        Removes leading and trailing spaces and converts string columns to lowercase except for URLs.
        :return: None
        """
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].apply(
                    lambda x: x.strip().lower().capitalize()
                    if isinstance(x, str) and not x.startswith('http') else x
                )

    def remove_by_column_values(self, column: str, value: list[str]) -> None:
        """
        Removes rows where the specified column has values in the given list.
        :param column: Column to filter.
        :param value: List of values to remove.
        :return: None
        """
        value_str = ', '.join(f"'{elem}'" for elem in value)
        self.df = self.df.query(f"{column} not in [{value_str}]")

    def handle_errors(self) -> None:
        """
        Handles erroneous values in numeric columns, replacing them with NaN
        :return: None
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def handle_missing_values(self) -> None:
        """
        Handles missing values in the DataFrame by filling them with appropriate defaults.
        :return: None
        """
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if set(self.df[col].dropna().unique()) == {True}:
                    self.df[col] = self.df[col].apply(lambda x: 1 if x is True else x).astype('Int64')  # .astype('int')
                # else:
                #     self.df[col] = self.df[col].fillna('Unknown')

            elif self.df[col].dtype in ['float64']:
                self.df[col] = self.df[col].apply(lambda x: int(x) if not pd.isna(x) else x).astype('Int64')
                # self.df[col] = self.df[col].fillna(self.df[col].mean()).astype(int)

    def replace_rare_values(self, column: str, replacement=pd.NA, min_amount: int = 20) -> None:
        """
        Replaces rare categories in the specified column with replacement parameter.
        :param replacement: Replace actual value for this (default is Nan)
        :param column: Column to process
        :param min_amount: Minimum count to keep category (default is 20)
        :return: None
        """
        value_counts = self.df[column].value_counts()
        rare_values = value_counts[value_counts < min_amount].index
        self.df[column] = self.df[column].apply(lambda x: replacement if x in rare_values else x)

    def write_to_csv(self, output_file: str, mode='w', sep=',', encoding='utf-8') -> None:
        """
        Saves the cleaned DataFrame to a CSV file.
        :param output_file: The path to the output file.
        :param mode: The write mode ('w' for overwriting, 'a' for appending) default is 'w'.
        :param sep: The delimiter to use between columns (default is ',').
        :param encoding: The encoding format to use for the output file (default is 'utf-8').
        :return: None
        """
        try:
            self.df.to_csv(output_file, mode=mode, sep=sep, encoding=encoding, index=False)
            print(f"File {output_file} created successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"Write path doesn't exist: {output_file}")
        except PermissionError:
            raise PermissionError(f"No permission to write to file: {output_file}")
        except Exception as e:
            raise Exception(f"Unknown error writing file {output_file}: {str(e)}")
