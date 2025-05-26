import pyodbc
import struct
from azure.identity import AzureCliCredential
from itertools import chain, repeat
import pandas as pd
import os

# 👉 Configura tu SQL endpoint y tabla
sql_endpoint = "4abiqltibpietbl7i3tuovxzoq-nmnholn43ire3nqnblz5l5ozma.datawarehouse.fabric.microsoft.com"
database     = "TestWarehouse"
table_names = [
    "Customers", "Products", "Stores", "Sales", "Employees",
    "Deliveries", "Suppliers", "Inventory", "Store_Issues", "Ratings"
]

# 🔐 Azure AD token via Azure CLI
credential = AzureCliCredential()
resource_url = "https://database.windows.net/.default"
access_token = credential.get_token(resource_url).token

# 🎯 Prepara el token para ODBC
token_bytes = access_token.encode("utf-8")
ex_token = bytes(chain.from_iterable(zip(token_bytes, repeat(0))))
token_struct = struct.pack("<i", len(ex_token)) + ex_token

# 🔌 Cadena de conexión a Microsoft Fabric SQL endpoint
connection_string = f"""
    Driver={{ODBC Driver 18 for SQL Server}};
    Server={sql_endpoint},1433;
    Database={database};
    Encrypt=yes;
    TrustServerCertificate=no;
"""

# 🚀 Ejecuta el SELECT vacío para obtener los encabezados
conn = pyodbc.connect(connection_string, attrs_before={1256: token_struct})
cursor = conn.cursor()
os.makedirs("parquet_data", exist_ok=True)

for table_name in table_names:
    # Ejecuta el SELECT vacío para obtener los encabezados
    cursor.execute(f"SELECT * FROM {table_name}")

    # Obtener y mostrar columnas
    columns = [col[0] for col in cursor.description]

    # Fetch all rows
    rows = cursor.fetchall()

    # Convertir a DataFrame
    df = pd.DataFrame.from_records(rows, columns=columns)

    # Guardar como Parquet
    parquet_filename = os.path.join("parquet_data", f"{table_name}.parquet")
    df.to_parquet(parquet_filename, index=False)
    print(f"✅ Data from '{table_name}' saved to '{parquet_filename}'")

# Cerrar conexión
cursor.close()
conn.close()