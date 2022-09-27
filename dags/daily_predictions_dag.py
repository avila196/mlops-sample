import datetime as dt
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

DAG_NAME = 'daily_predictions'
MODEL_ENDPOINT_PREDICTION = "http://tc:5000/predict"
FILES_LOCATION_PREDICTION = "record_files"

args = {
    'owner': 'david.avila',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'sla': dt.timedelta(minutes=30),
    'execution_timeout': dt.timedelta(minutes=25),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1)
}

def request_predictions(model_endpoint, files_location):
    """
    Requests the predictions endpoint of the model container
    """
    response = requests.post(model_endpoint, data={"location": files_location})
    return response.content.decode("utf8")

def perform_predictions(**context):
    """
    Call model container endpoint according to parameters and returns the final message
    """
    #We get the model endpoint and files location, we perform the request and we return back the result
    model_endpoint = context['templates_dict']['model_endpoint']
    files_location = context['templates_dict']['files_location']
    message = request_predictions(model_endpoint, files_location)
    return message

with DAG(
    dag_id=DAG_NAME,
    schedule_interval="@hourly",
    max_active_runs=1,
    default_args=args,
    tags=['fintech-classification'],
) as dag:

    # First task: Retrieve new files
    # NOTE: This step is assumed already given that there are no credentials being checked to access S3 buckets
    # or similar. For testing, the files already reside within the model container. So, a DummyOperator is created
    # to show the retrieval is done
    retrieval_done_task = DummyOperator(task_id='retrieval_task', dag=dag)

    # Second task: Trigger predictions job.
    # Instead of calling a SparkSubmitOperator, we're using a PythonOperator to request the model prediction endpoint
    # to simulate the Spark job, and we get back the results (they get printed/logged within Airflow Logs).
    perform_predictions = PythonOperator(
        task_id=f"perform_predictions_task",
        provide_context=True,
        templates_dict={"model_endpoint": MODEL_ENDPOINT_PREDICTION,
                        "files_location": FILES_LOCATION_PREDICTION},
        python_callable=perform_predictions,
        dag=dag
    )

    # Third task: Export results into database
    # As the Redshift data warehouse is not set, we're just going to create another DummyOperator that says that all
    # results were correctly exported.
    export_task = DummyOperator(task_id='exported_predictions_task', dag=dag)

    # Order of tasks
    retrieval_done_task >> perform_predictions >> export_task