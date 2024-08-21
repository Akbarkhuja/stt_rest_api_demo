import json
import requests
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, lit
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, MapType
from pyspark.sql import Row


spark = SparkSession.builder.appName("STT_PIPELINE").getOrCreate()


schema = StructType([
    StructField('audio_file_path', StringType(), True),
    StructField('request_id', StringType(), True),
    StructField('text', StringType(), True),
])


def send_post_request(audio_file_path): # sending audio file in a byte format
    url = "https://api.edenai.run/v2/audio/speech_to_text_async"
    headers = {
        "Authorization": 
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiOTk2MDg1MmQtZjEzZS00MWViLWJmZWEtZTAwYjNhMGY3NWM1IiwidHlwZSI6ImFwaV90b2tlbiJ9.y8ScqrWHizK013XpUxSb8PScSJWmI6OTITGJn3VHwdo"
    }

    data = {
        "providers": "google",       
        "language": "en-US",
    }
    files = {'file': open(audio_file_path, 'rb')}

    response = requests.post(url, data=data, files=files, headers=headers)
    result = json.loads(response.text)
    return result


def send_get_request(id): # sending id - key to get text of the audio file
    url = "https://api.edenai.run/v2/audio/speech_to_text_async"
    headers = {
        "Authorization": 
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiOTk2MDg1MmQtZjEzZS00MWViLWJmZWEtZTAwYjNhMGY3NWM1IiwidHlwZSI6ImFwaV90b2tlbiJ9.y8ScqrWHizK013XpUxSb8PScSJWmI6OTITGJn3VHwdo"
    }
    
    text_response = requests.get(url + '/' + id, headers=headers)
    json_text_response = json.loads(text_response.text)

    while json_text_response['results']['google']['final_status'] == 'processing':
        time.sleep(10)
        if json_text_response['results']['google']['error'] != None:
            return ''
        
        text_response = requests.get(url + '/' + id, headers=headers)
        json_text_response = json.loads(text_response.text)
    
    print("JSON FORMAT: ")
    print(type(json_text_response))
    return json_text_response['results']['google']['text']

def audio_to_text(audio_file_path):
    audio_response = send_post_request(audio_file_path)
    public_id = audio_response['public_id']

    text_response = send_get_request(public_id)
    
    result = {
        # 'audio_file_path': audio_file_path,
        'request_id': public_id,
        'text': text_response
    }
    return result

# AUDIO File Path
audio_file_path = 'audio_files/harvard.wav'

# User Defined Fuction (UDF) Declaration
udf_audio_to_text = udf(audio_to_text, schema)

# Dataframe Creation
RestApiRequestRow = Row('audio_file_path')
requests_df = spark.createDataFrame([
    RestApiRequestRow(audio_file_path)
])

# Appending the Rest API UDF 
requests_df = requests_df.withColumn(
    "result", udf_audio_to_text(col('audio_file_path'))
)


df = requests_df.select(explode(col('result.text')).alias("text")) # Error --> data type mismatch
df.select(collapse_columns(df.schema)).show()


