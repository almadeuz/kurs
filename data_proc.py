import pandas as pd
import numpy as np
from typing import Optional

def load_data() -> Optional[pd.DataFrame]:
    """Загрузка и обработка данных"""

    data_df = pd.read_csv("diabetes.csv")
    
    repl_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for feature in repl_zeros:
        if feature in data_df.columns:
            data_df[feature] = data_df[feature].replace(0, np.nan)
            median_0 = data_df[data_df["Outcome"] == 0][feature].median()
            median_1 = data_df[data_df["Outcome"] == 1][feature].median()
            data_df.loc[(data_df["Outcome"] == 0) & (data_df[feature].isnull()), feature] = median_0
            data_df.loc[(data_df["Outcome"] == 1) & (data_df[feature].isnull()), feature] = median_1
    
    data_df.fillna(data_df.median(), inplace=True)

    return data_df
