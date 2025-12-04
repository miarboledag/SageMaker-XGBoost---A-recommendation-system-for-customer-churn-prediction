import json
import boto3
import io
import csv
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = "churn-xgboost-prod-v1" 
TARGET_CHURN_PROBABILITY_CUTOFF = 0.32 

# --- NOMBRES DE COLUMNA (100 columnas: Target + 99 Features) ---
ALL_COLUMNS_100 = ['Churn?_True.', 'Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Eve Mins', 'Eve Calls', 'Night Mins', 'Night Calls', 'Intl Mins', 'Intl Calls', 'CustServ Calls', 'State_AK', 'State_AL', 'State_AR', 'State_AZ', 'State_CA', 'State_CO', 'State_CT', 'State_DC', 'State_DE', 'State_FL', 'State_GA', 'State_HI', 'State_IA', 'State_ID', 'State_IL', 'State_IN', 'State_KS', 'State_KY', 'State_LA', 'State_MA', 'State_MD', 'State_ME', 'State_MI', 'State_MN', 'State_MO', 'State_MS', 'State_MT', 'State_NC', 'State_ND', 'State_NE', 'State_NH', 'State_NJ', 'State_NM', 'State_NV', 'State_NY', 'State_OH', 'State_OK', 'State_OR', 'State_PA', 'State_RI', 'State_SC', 'State_SD', 'State_TN', 'State_TX', 'State_UT', 'State_VA', 'State_VT', 'State_WA', 'State_WI', 'State_WV', 'State_WY', 'Area Code_657', 'Area Code_658', 'Area Code_659', 'Area Code_676', 'Area Code_677', 'Area Code_678', 'Area Code_686', 'Area Code_707', 'Area Code_716', 'Area Code_727', 'Area Code_736', 'Area Code_737', 'Area Code_758', 'Area Code_766', 'Area Code_776', 'Area Code_777', 'Area Code_778', 'Area Code_786', 'Area Code_787', 'Area Code_788', 'Area Code_797', 'Area Code_798', 'Area Code_806', 'Area Code_827', 'Area Code_836', 'Area Code_847', 'Area Code_848', 'Area Code_858', 'Area Code_866', 'Area Code_868', 'Area Code_876', 'Area Code_877', 'Area Code_878', "Int'l Plan_no", "Int'l Plan_yes", 'VMail Plan_no', 'VMail Plan_yes']

# --- REGLAS DE NEGOCIO (Motor de Recomendación) ---
def motor_de_recomendacion(fila):
    """
    Analiza las columnas de features preprocesadas para dar una recomendación.
    """
    # Regla 1: Cliente frustrado (muchas llamadas a soporte)
    if fila['CustServ Calls'] > 3:
        return "Trato VIP: Asignar gestor personal + Prioridad en atención"
    
    # Regla 2: Cliente que gasta mucho (VIP Económico)
    minutos_total = fila['Day Mins'] + fila['Eve Mins'] + fila['Night Mins'] + fila['Intl Mins']
    if minutos_total > 500:
        return "Retención Financiera: 25% de descuento en la factura x 6 meses"
    
    # Regla 3: Usuario Internacional (Viajero o familia fuera)
    if fila['Intl Calls'] > 4 or fila["Int'l Plan_yes"] == 1:
        return "Pack Viajero: Bonificación de 200 minutos internacionales"
    
    # Regla 4: Usuario de Datos/Voz intensivo (Habla mucho de día)
    if fila['Day Mins'] > 220:
        return "Upgrade Tecnológico: Oferta de renovación de equipo (Smartphone)"

    # Regla por defecto
    return "Incentivo General: 3 meses de Amazon Prime Video gratis"

# --- HANDLER PRINCIPAL ---
def lambda_handler(event, context):
    
    csv_string = event.get("body", "")
    
    if event.get("isBase64Encoded"):
        import base64
        csv_string = base64.b64decode(csv_string).decode("utf-8")

    # 1. Leer el CSV en un DataFrame
    try:
        df_full = pd.read_csv(io.StringIO(csv_string), header=None, names=ALL_COLUMNS_100)
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": f"Error al leer el CSV: {str(e)}"})
        }

    # 2. AÑADIR COLUMNA DE ÍNDICE ORIGINAL (Fila en CSV)
    # Esto es clave: guardamos el número de fila original (comenzando desde 1 como Excel)
    df_full['Original_Row_Number'] = range(1, len(df_full) + 1)

    # 3. Extraer las 99 features para SageMaker
    sagemaker_input_data = df_full.iloc[:, 1:100].to_numpy(dtype=float)  # Columnas 1-99 (excluye target)

    # 4. Invocar SageMaker
    all_preds = []
    for row_batch in np.array_split(sagemaker_input_data, max(1, int(sagemaker_input_data.shape[0] / 500.0) + 1)):
        payload = pd.DataFrame(row_batch).to_csv(header=False, index=False)
        
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )

        result = response["Body"].read().decode("utf-8").strip()
        preds = [float(x) for x in result.split("\n") if x]
        all_preds.extend(preds)

    # 5. Generar Reporte
    df_full['Probabilidad_Fuga'] = all_preds
    df_full['Riesgo_Alto'] = df_full['Probabilidad_Fuga'] > TARGET_CHURN_PROBABILITY_CUTOFF
    df_riesgo = df_full[df_full['Riesgo_Alto']].copy()
    
    # Aplicar recomendaciones
    df_riesgo['Accion_Recomendada'] = df_riesgo.apply(motor_de_recomendacion, axis=1)

    # 6. Formatear la Salida JSON
    # Seleccionamos columnas clave INCLUYENDO el número de fila original
    columnas_reporte = ['Original_Row_Number', 'Account Length', 'CustServ Calls', 'Day Mins', 'Intl Calls', 'Probabilidad_Fuga', 'Accion_Recomendada']
    
    # Crear lista de diccionarios con los datos
    clientes_en_riesgo = []
    for _, row in df_riesgo.iterrows():
        cliente = {
            'Fila_CSV_Original': int(row['Original_Row_Number']),  # ¡ESTO ES LO NUEVO Y CLAVE!
            'Account Length': float(row['Account Length']),
            'CustServ Calls': int(row['CustServ Calls']),
            'Day Mins': float(row['Day Mins']),
            'Intl Calls': int(row['Intl Calls']),
            'Probabilidad_Fuga': float(row['Probabilidad_Fuga']),
            'Accion_Recomendada': row['Accion_Recomendada']
        }
        clientes_en_riesgo.append(cliente)

    # 7. Respuesta final para el Frontend
    resp = {
        "rows_processed": len(df_full),
        "total_clientes_riesgo": len(df_riesgo),
        "clientes_en_riesgo": clientes_en_riesgo,
        "mensaje": f"Análisis completado. {len(df_riesgo)} cliente(s) identificado(s) con riesgo > {TARGET_CHURN_PROBABILITY_CUTOFF}.",
        # Información adicional útil
        "metadata": {
            "total_filas_csv": len(df_full),
            "umbral_riesgo": TARGET_CHURN_PROBABILITY_CUTOFF,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(resp, default=str)
    }