import pandas as pd
import os
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import load_model

class RainfallDataset:
    def __init__(self, data_path:str = None):
        self.data_path = data_path
        self.normalization_params = None
        self.temporal_windows = None
        self.data: pd.DataFrame = None
        self.inflow: pd.DataFrame = None
        self.outflow: pd.DataFrame = None
        self.train_data = None
        self.train_inflow = None
        self.train_outflow = None
        self.val_data = None
        self.val_inflow = None
        self.val_outflow = None
        self.test_data = None
        self.test_inflow = None
        self.test_outflow = None
        self.data_scaler = MinMaxScaler()
        self.inflow_scaler = MinMaxScaler()
        self.outflow_scaler = MinMaxScaler()
        self.train_data_norm = None
        self.val_data_norm = None
        self.test_data_norm = None
        self.train_inflow_norm = None
        self.val_inflow_norm = None
        self.test_inflow_norm = None
        self.train_outflow_norm = None
        self.val_outflow_norm = None
        self.test_outflow_norm = None
        self.model = None

    def load_excel(self, data_path:str, temporal_column:str=None)->None:
        """
        Load data from an Excel file and add it to the existing DataFrame.

        Parameters:
        - data_path (str): Path to the Excel file.
        - temporal_column (str, optional): Name of the temporal column. Defaults to 'date'.

        This method reads data from an Excel file, drops columns with all null values, renames the temporal column,
        converts the 'date' column to datetime format, merges dataframes if data already exists, cleans the data by
        removing null values, replaces negative values with zero, sets 'date' as index, and splits the dataset into
        train, validation, and test sets.
        """
        new_data = pd.read_excel(data_path)
        new_data = new_data.dropna(axis=1, how='all')
        if temporal_column is None:
            temporal_column = 'date'
        new_data = new_data.rename(columns={temporal_column: temporal_column})
        new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')
        if self.data is None:
            self.data = new_data
        else:
            self.data = pd.merge(self.data, new_data, how='inner', on='date')
        self.clean_data()
        self.data = self.replace_negatives_with_zero(df=self.data)
        self.data = self.data.set_index('date').reset_index()
        self.train_val_test()

    def load_csv(self, data_path:str, temporal_column:str=None)->None:
        """
        Load data from a csv file and add it to the existing DataFrame.

        Parameters:
        - data_path (str): Path to the csv file.
        - temporal_column (str, optional): Name of the temporal column. Defaults to 'date'.

        This method reads data from an csv file, drops columns with all null values, renames the temporal column,
        converts the 'date' column to datetime format, merges dataframes if data already exists, cleans the data by
        removing null values, replaces negative values with zero, sets 'date' as index, and splits the dataset into
        train, validation, and test sets.
        """
        new_data = pd.read_csv(data_path)
        if temporal_column is None:
            temporal_column = 'date'
        new_data = new_data.rename(columns={temporal_column: temporal_column})
        new_data['date'] = pd.to_datetime(new_data[temporal_column], errors='coerce')
        if self.data is None:
            self.data = new_data
        else:
            self.data = pd.merge(self.data, new_data, how='inner', on='date')
        self.clean_data()
        self.data = self.replace_negatives_with_zero(df=self.data)
        self.data = self.data.set_index('date').reset_index()
        self.train_val_test()

    def load_data(self, data_path:str, temporal_column:str)->None:
        """
        Load data from a specified file based on its extension and add it to the existing DataFrame.

        Parameters:
        - data_path (str): Path to the data file.
        - temporal_column (str): Name of the temporal column.

        This method delegates the loading of data to specific methods based on the file extension.
        If the file is a CSV, it calls 'load_csv'; if it is an Excel file, it calls 'load_excel'.
        Unsupported file extensions result in a printed warning.
        """
        if data_path.endswith('.csv'):
            self.load_csv(data_path=data_path, temporal_column=temporal_column)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            self.load_excel(data_path=data_path, temporal_column=temporal_column)
        else:
            print("File extension not supported")


    def load_folder(self, data_path:str,temporal_column:str)->None:
        """
        Load data from multiple files within a specified folder and add it to the existing DataFrame.

        Parameters:
        - data_path (str): Path to the folder containing data files.
        - temporal_column (str): Name of the temporal column.

        This method iterates over all files in the specified folder, calling 'load_data' for each file.
        """
        for filename in os.listdir(data_path):
            self.load_data(data_path = data_path+filename,temporal_column=temporal_column)
    
    def clean_data(self)->None:
        """
        Remove rows with null values from the dataset.

        This method drops rows containing any null or NaN values in any column.
        """
        self.data = self.data.dropna()

    def set_inflow(self, inflow_column:str)->None:
        """
        Set the inflow column for the dataset and update the train, validation, and test sets.

        Parameters:
        - inflow_column (str): Name of the inflow column.

        If the specified inflow column is valid, it sets the inflow attribute, and then updates the
        train, validation, and test sets using 'train_val_test'.
        Otherwise, it prints a warning.
        """
        if inflow_column in self.data.columns:
            self.inflow = self.data[['date',inflow_column]]
            self.train_val_test()

        else:
            print(f'{inflow_column} is not a valid column name')

    def set_outflow(self, outflow_column:str)->None:
        """
        Set the outflow column for the dataset, update the train, validation, and test sets,
        and normalize the data.

        Parameters:
        - outflow_column (str): Name of the outflow column.

        If the specified outflow column is valid, it sets the outflow attribute, updates the
        train, validation, and test sets using 'train_val_test', and normalizes the data using
        'normalization'.
        Otherwise, it prints a warning.
        """
        if outflow_column in self.data.columns:
            self.outflow = self.data[['date',outflow_column]]
            self.train_val_test()
            self.normalization()
        else:
            print(f'{outflow_column} is not a valid column name')
    
    def replace_negatives_with_zero(self, df) -> None:
        """
        Replace negative values with zero in the specified DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing numerical and datetime columns.

        Returns:
        pd.DataFrame: A new DataFrame with negative values replaced by zero.

        This method creates a copy of the input DataFrame and replaces negative values with zero
        in the selected numerical and datetime columns.
        """
        df_copy = df.copy()
        numeric_columns = df_copy.select_dtypes(include=['float64', 'int64', 'object']).columns
        numeric_columns = [col for col in numeric_columns if pd.to_numeric(df_copy[col], errors='coerce').notnull().all()]
        date_columns = df_copy.select_dtypes(include=['datetime64']).columns
        selected_columns = list(set(numeric_columns).union(set(date_columns)))
        df_copy = df_copy[selected_columns]
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(lambda x: 0 if (pd.to_numeric(x, errors='coerce') < 0) else x)

        return df_copy.sort_index(axis=1)

    
    def train_val_test(self,val_year:int=None)->None:
        """
        Split the dataset into training, validation, and test sets based on the specified validation year.

        Parameters:
        - val_year (int, optional): The year to use for validation. If None, the second-to-last year is used for validation.

        Returns:
        None

        This method divides the dataset into training, validation, and test sets based on the specified or default validation year.
        It also normalizes the datasets.
        """
        unique_years = self.data['date'].dt.year.unique()
        if val_year is None:
            self.val_year = [unique_years[-2]]
            self.test_year = [unique_years[-1]]
            self.train_year = [year for year in unique_years if year not in [self.test_year,self.val_year]]

        self.train_data = self.data[self.data['date'].dt.year.isin(self.train_year)]
        self.val_data = self.data[self.data['date'].dt.year.isin(self.val_year)]
        self.test_data = self.data[self.data['date'].dt.year.isin(self.test_year)]

        if self.inflow is not None:
            self.train_inflow = self.inflow[self.inflow['date'].dt.year.isin(self.train_year)]
            self.val_inflow = self.inflow[self.inflow['date'].dt.year.isin(self.val_year)]
            self.test_inflow = self.inflow[self.inflow['date'].dt.year.isin(self.test_year)]

        if self.outflow is not None:
            self.train_outflow = self.inflow[self.inflow['date'].dt.year.isin(self.train_year)]
            self.val_outflow = self.inflow[self.inflow['date'].dt.year.isin(self.val_year)]
            self.test_outflow = self.inflow[self.inflow['date'].dt.year.isin(self.test_year)]
        self.normalization()

    def save_scaler(self, folder='scaler', scaler_name='train_scaler', scaler=None):
        """
        Save the scaler to a file.

        Parameters:
        - folder (str, optional): The folder path where the scaler file will be saved.
        - scaler_name (str, optional): The name of the scaler file.
        - scaler (Any, optional): The scaler object to save. If None, the dataset scaler is used.

        Returns:
        None

        This method saves the scaler to a file with the specified name and in the specified folder.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        scaler_path = os.path.join(folder, f"{scaler_name}.pkl")
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
        if scaler is None:
            joblib.dump(self.data_scaler, scaler_path)
        else:
            joblib.dump(scaler, scaler_path)

    def normalization(self)->None:
        """
        Normalize the training, validation, and test datasets and save the scalers.

        Returns:
        None

        This method normalizes the datasets and saves the scalers for the features, inflow, and outflow.
        """
        train_data = self.train_data.drop(columns=['date'])
        val_data = self.val_data.drop(columns=['date'])
        test_data = self.test_data.drop(columns=['date'])
        self.train_data_norm = pd.DataFrame(self.data_scaler.fit_transform(train_data), columns=train_data.columns)
        self.val_data_norm = pd.DataFrame(self.data_scaler.transform(val_data), columns=val_data.columns)
        self.test_data_norm = pd.DataFrame(self.data_scaler.transform(test_data), columns=test_data.columns)
        self.train_data_norm['date'] = self.train_data['date']
        self.val_data_norm['date'] = self.val_data['date']
        self.test_data_norm['date'] = self.test_data['date']

        if self.inflow is not None:
            train_inflow = self.train_inflow.drop(columns=['date'])
            val_inflow = self.val_inflow.drop(columns=['date'])
            test_inflow = self.test_inflow.drop(columns=['date'])
            self.train_inflow_norm = pd.DataFrame(self.inflow_scaler.fit_transform(train_inflow), columns=train_inflow.columns)
            self.val_inflow_norm = pd.DataFrame(self.inflow_scaler.transform(val_inflow), columns=val_inflow.columns)
            self.test_inflow_norm = pd.DataFrame(self.inflow_scaler.transform(test_inflow), columns=test_inflow.columns)
            self.train_inflow_norm['date'] = self.train_inflow['date']
            self.val_inflow_norm['date'] = self.val_inflow['date']
            self.test_inflow_norm['date'] = self.test_inflow['date']

        if self.outflow is not None:
            train_outflow = self.train_outflow.drop(columns=['date'])
            val_outflow = self.val_outflow.drop(columns=['date'])
            test_outflow = self.test_outflow.drop(columns=['date'])
            self.train_outflow_norm = pd.DataFrame(self.outflow_scaler.fit_transform(train_outflow), columns=train_outflow.columns)
            self.val_outflow_norm = pd.DataFrame(self.outflow_scaler.transform(val_outflow), columns=val_outflow.columns)
            self.test_outflow_norm = pd.DataFrame(self.outflow_scaler.transform(test_outflow), columns=test_outflow.columns)
            self.train_outflow_norm['date'] = self.train_outflow['date']
            self.val_outflow_norm['date'] = self.val_outflow['date']
            self.test_outflow_norm['date'] = self.test_outflow['date']
        self.save_scaler(scaler_name='x_train_scaler',scaler=self.data_scaler)
        self.save_scaler(scaler_name='inflow_train_scaler',scaler=self.inflow_scaler)
        self.save_scaler(scaler_name='outflow_train_scaler',scaler=self.outflow_scaler)
    

  


    def __str__(self)->None:
        """
        Return a formatted string containing information about the dataset, inflow, and outflow.

        Returns:
            str: Formatted string with dataset, inflow, and outflow information.
        """
        data_table = PrettyTable()
        inflow_table = PrettyTable()
        outflow_table = PrettyTable()
        data_table.field_names = ["Data", "Value"]
        inflow_table.field_names = ["Inflow", "Value"]
        outflow_table.field_names = ["Outflow", "Value"]
        data_table.add_row(["Columns", ', '.join(self.data.columns)])
        data_table.add_row(["Number of rows", len(self.data)])
        data_table.add_row(["Train years", self.train_year])
        data_table.add_row(["Val year", self.val_year])
        data_table.add_row(["Test years", self.test_year])
        data_table.add_row(["Null values", self.data.isnull().sum().sum()])  # Add the number of null values
        data_describe = self.data.drop(columns='date').describe() if 'date' in self.data.columns else self.data.describe()
        data_table.add_row(["Summary", data_describe])

        if self.inflow is not None:
            inflow_table.add_row(["Columns", ', '.join(self.inflow.columns)])
            inflow_table.add_row(["Number of rows", len(self.inflow)])
            inflow_table.add_row(["Null values", self.inflow.isnull().sum().sum()])  # Add the number of null values
            inflow_describe = self.inflow.drop(columns='date').describe() if 'date' in self.inflow.columns else self.inflow.describe()
            inflow_table.add_row(["Summary", inflow_describe])

        if self.outflow is not None:
            outflow_table.add_row(["Columns", ', '.join(self.outflow.columns)])
            outflow_table.add_row(["Number of rows", len(self.outflow)])
            outflow_table.add_row(["Null values", self.outflow.isnull().sum().sum()])  # Add the number of null values
            outflow_describe = self.outflow.drop(columns='date').describe() if 'date' in self.outflow.columns else self.outflow.describe()
            outflow_table.add_row(["Summary", outflow_describe])

        # Return the tables as a string
        return (f'\n\n{"*"*50} DATASET INFO {"*"*50}\n\n'
                f'{data_table}\n\n'
                f'{inflow_table}\n\n'
                f'{outflow_table}\n\n')
    

class PredictionDataset(RainfallDataset):
    def __init__(self, data_path=None):
        super().__init__(data_path)

    def load_excel(self, data_path:str, temporal_column:str=None)->None:
        """
        Load data from an Excel file into the dataset.

        Args:
            data_path (str): Path to the Excel file.
            temporal_column (str, optional): Name of the temporal column. Default is None.

        Returns:
            None
        """
        new_data = pd.read_excel(data_path)
        new_data = new_data.dropna(axis=1, how='all')
        if temporal_column is None:
            temporal_column = 'date'
        new_data = new_data.rename(columns={temporal_column: temporal_column})
        new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')
        if self.data is None:
            self.data = new_data
        else:
            self.data = pd.merge(self.data, new_data, how='outer', on='date')
        self.clean_data()
        self.data = self.replace_negatives_with_zero(df=self.data)
        self.data = self.data.set_index('date').reset_index()

    def load_csv(self, data_path:str, temporal_column:str=None)->None:
        """
        Load data from an csv file into the dataset.

        Args:
            data_path (str): Path to the csv file.
            temporal_column (str, optional): Name of the temporal column. Default is None.

        Returns:
            None
        """
        new_data = pd.read_csv(data_path)
        if temporal_column is None:
            temporal_column = 'date'
        new_data = new_data.rename(columns={temporal_column: temporal_column})
        new_data['date'] = pd.to_datetime(new_data[temporal_column], errors='coerce')
        if self.data is None:
            self.data = new_data
        else:
            self.data = pd.merge(self.data, new_data, how='outer', on='date')
        self.clean_data()
        self.data = self.replace_negatives_with_zero(df=self.data)
        self.data = self.data.set_index('date').reset_index()

    def set_inflow(self, inflow_column:str)->None:
        """
        Set the inflow data for the dataset based on the specified column.

        Args:
            inflow_column (str): Name of the column containing inflow data.

        Returns:
            None
        """
        if inflow_column in self.data.columns:
            self.inflow = self.data[['date',inflow_column]]
        else:
            print(f'{inflow_column} is not a valid column name')

    def set_outflow(self, outflow_column:str)->None:
        """
        Set the outflow data for the dataset based on the specified column.

        Args:
            outflow_column (str): Name of the column containing outflow data.

        Returns:
            None
        """
        if outflow_column in self.data.columns:
            self.outflow = self.data[['date',outflow_column]]
        else:
            print(f'{outflow_column} is not a valid column name')
    
    
    
def load_keras_model(model_path):
    """
    Load a Keras model from the specified path.

    Parameters:
    - model_path (str): Path to the Keras model file (.keras).

    Returns:
    - model: Loaded Keras model.
    """
    model = load_model(model_path)
    return model

def load_scaler(scaler_path):
    """
    Load a scaler from the specified path using joblib.

    Parameters:
    - scaler_path (str): Path to the scaler file (.pkl).

    Returns:
    - scaler: Loaded scaler.
    """
    scaler = joblib.load(scaler_path)
    return scaler

if __name__ == "__main__":
    
    path_dataset = 'datasets'
    folder_path = 'datasets/csv_folder/'
    excel_path = 'datasets/csv_folder/'
    arzua_path = os.path.join('datasets','arzua.cxv')
    output_path = os.path.join(excel_path,'outputs.xlsx')
    portodemouros = RainfallDataset()
    portodemouros.load_folder(data_path=folder_path,temporal_column='date')
    portodemouros.set_inflow(inflow_column='Inflow')
    portodemouros.set_outflow(outflow_column='Outflow')
    print(portodemouros)