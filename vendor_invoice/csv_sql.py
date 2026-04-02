import os
import pandas as pd
import mysql.connector
from mysql.connector import Error

# -------------------------------
# 1. DATABASE CONNECTION
# -------------------------------
def create_connection(host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            ssl_disabled=True  # 🔒 disable SSL for local connections
        )
        if conn.is_connected():
            print("✅ Connected to MySQL")
        return conn
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None


# -------------------------------
# 2. CREATE TABLE FROM CSV
# -------------------------------
def create_table_from_csv(cursor, table_name, df):
    """
    Dynamically create MySQL table using CSV columns
    """
    columns = []

    for col, dtype in zip(df.columns, df.dtypes):
        if "int" in str(dtype):
            mysql_type = "INT"
        elif "float" in str(dtype):
            mysql_type = "FLOAT"
        elif "datetime" in str(dtype):
            mysql_type = "DATETIME"
        else:
            mysql_type = "TEXT"
        
        columns.append(f"`{col}` {mysql_type}")

    columns_sql = ", ".join(columns)
    create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({columns_sql});"

    cursor.execute(create_table_query)
    print(f"📌 Table `{table_name}` ready.")


# -------------------------------
# 3. INSERT CSV DATA INTO MYSQL
# -------------------------------
def insert_data(cursor, conn, table_name, df):

    placeholders = ", ".join(["%s"] * len(df.columns))
    columns = ", ".join([f"`{col}`" for col in df.columns])

    insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"

    df = df.fillna("")

    chunk_size = 500   # 🔥 smaller chunk prevents crash

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        data = [tuple(row) for row in chunk.values]

        try:
            cursor.executemany(insert_query, data)
            conn.commit()
            print(f"Inserted rows {i} to {i+chunk_size}")

        except mysql.connector.Error as e:
            print("⚠️ reconnecting MySQL...")
            conn.reconnect(attempts=3, delay=5)
            cursor = conn.cursor()
            cursor.executemany(insert_query, data)
            conn.commit()



# -------------------------------
# 4. LOAD ALL CSV FILES FROM FOLDER
# -------------------------------
def load_multiple_csv_to_mysql(folder_path, conn):
    cursor = conn.cursor()

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            table_name = file.replace(".csv", "")

            print(f"\n📂 Processing file: {file}")

            df = pd.read_csv(file_path)

            create_table_from_csv(cursor, table_name, df)
            insert_data(cursor, conn, table_name, df)

    cursor.close()


# -------------------------------
# 5. MAIN FUNCTION
# -------------------------------
if __name__ == "__main__":

    # MySQL Credentials
    HOST = "localhost"
    USER = "root"
    PASSWORD = "123456"
    DATABASE = "ml"


    # CSV Folder Path
    CSV_FOLDER = r"G:\ML_projects\vendor_invoice"

    # Connect MySQL
    connection = create_connection(HOST, USER, PASSWORD, DATABASE)

    if connection:
        load_multiple_csv_to_mysql(CSV_FOLDER, connection)
        connection.close()
        print("\n🎉 All CSV files loaded successfully!")
