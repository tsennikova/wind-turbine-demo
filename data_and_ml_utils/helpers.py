# Databricks notebook source
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

!pip install windrose

# COMMAND ----------



# COMMAND ----------

aws_bucket_name = "oetrta/tania"
file_type = "csv"
mount_name = "/tania/wind-turbine-demo"
try:
  dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)
except:
  dbutils.fs.unmount("/mnt/%s" % mount_name)
  dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)
#display(dbutils.fs.ls("/mnt/%s" % mount_name))

# COMMAND ----------

def get_notebook_tag(tag):
    tag = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag)
    return None if tag.isEmpty() else tag.get()


# COMMAND ----------

user = get_notebook_tag("user")
workspace_home_dir = f"/Users/{user}"
base_dir_fuse = f"/dbfs/tmp/{user}/mlflow_demo/"

notebook_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
experiment_name = notebook_name 

# COMMAND ----------

import os
import requests

def download_file(data_uri, data_path):
    if os.path.exists(data_path):
        print("File {} already exists".format(data_path))
    else:
        print("Downloading {} to {}".format(data_uri,data_path))
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        rsp = requests.get(data_uri)
        with open(data_path, 'w') as f:
            f.write(rsp.text)

# COMMAND ----------

def download_data():
    data_path = f"{base_dir_fuse}/wine-quality.csv"
    data_uri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    download_file(data_uri, data_path)    
    return data_path, "quality"
#   return "/dbfs/databricks-datasets/flights/departuredelays.csv","delay"
#   return "/dbfs/databricks-datasets/atlas_higgs/atlas_higgs.csv", "Weight"

# COMMAND ----------

import mlflow
import pandas as pd
def predict_sklearn(model_uri, data):  
  model = mlflow.sklearn.load_model(model_uri)
  predictions = model.predict(data)
  df = pd.DataFrame(predictions,columns=["prediction"])
  return df

# COMMAND ----------

def predict_python(model_uri, data):  
  model = mlflow.pyfunc.load_model(model_uri)
  predictions = model.predict(data) >0.5
  df = pd.DataFrame(predictions,columns=["is_high_quality"])
  return df

# COMMAND ----------

def predict_spark(model_uri, data):  
  input = spark.createDataFrame(data)
  udf = mlflow.pyfunc.spark_udf(spark, model_uri)
  predictions = input.withColumn("is_high_quality", udf(*input.columns) > 0.5).select("is_high_quality")
  return predictions

# COMMAND ----------

def show_boxplot(data,target_column):
  import seaborn as sns
  import matplotlib.pyplot as plt
  dims = (3, 4)
  f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
  axis_i, axis_j = 0, 0
  for col in data.columns:
    if col == target_column:
      continue 
    sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
    axis_j += 1
    if axis_j == dims[1]:
      axis_i += 1
      axis_j = 0

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

def get_experiment_id(experiment_name):  
  return client.get_experiment_by_name(experiment_name).experiment_id
  

# COMMAND ----------

def clear_runs(experiment_name):  
  experiment_id = get_experiment_id(experiment_name)
  runs = client.list_run_infos(experiment_id)
  for info in runs:
    client.delete_run(info.run_id)
  

# COMMAND ----------


