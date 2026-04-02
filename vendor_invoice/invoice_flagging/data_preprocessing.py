# data_preprocessing.py
from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/ml')

    merge_query = """
    WITH purchase_agg AS (
        SELECT 
            PONumber,
            COUNT(DISTINCT Brand) AS total_brand,
            SUM(Quantity) AS total_qty,
            SUM(Dollars) AS total_dollars,
            ROUND(AVG(DATEDIFF(ReceivingDate, PODate)), 2) AS avg_lead_time
        FROM purchases
        GROUP BY PONumber
    )

    SELECT 
        v.PONumber,
        v.Quantity,
        v.Dollars,
        v.Freight,
        ROUND(DATEDIFF(v.InvoiceDate, v.PODate), 2) AS day_po_to_invoice,
        ROUND(DATEDIFF(v.PayDate, v.InvoiceDate), 2) AS day_to_pay,
        pa.total_brand,
        pa.total_qty,
        pa.total_dollars,
        pa.avg_lead_time
    FROM vendor_invoice v
    LEFT JOIN purchase_agg pa
    ON v.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(merge_query, engine)
    return df


def preprocess_data(df):
    features = df[[
        'total_brand',
        'total_qty',
        'total_dollars',
        'avg_lead_time',
        'day_po_to_invoice',
        'day_to_pay'
    ]]

    features = features.fillna(features.mean())

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler, df