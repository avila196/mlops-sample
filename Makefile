airflow-init:
	docker-compose up airflow-init

main-containers:
	docker-compose up

all: airflow-init main-containers