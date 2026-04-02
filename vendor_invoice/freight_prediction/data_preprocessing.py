from sqlalchemy import create_engine , text
import pandas as pd
from sklearn.model_selection import train_test_split

def load_vendor_data():

    # load data from mysql database
    engine = create_engine(f"mysql+mysqlconnector://root:123456@localhost:3306/ml")
    query = text("SELECT * FROM vendor_invoice")
    df = pd.read_sql_query(query, engine)
    return df

def prepare_feature(df : pd.DataFrame):

    # select feature and targwt variable
    x = df[['Dollars']]
    y = df['Freight']
    return x , y

def split_data(x , y , test_size = 0.2 , random_state = 42):

    # train test split
    return train_test_split(x , y , test_size = test_size , random_state = random_state)