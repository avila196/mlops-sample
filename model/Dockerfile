FROM python:3.7

WORKDIR /docker

COPY . /docker

RUN pip install -r requirements.txt && \
    curl -o /docker/app/topics_classifier_latest.h5 https://classifier-files-fintech.s3.us-east-2.amazonaws.com/topics_classifier_latest.h5

WORKDIR /docker/app

EXPOSE 5000

CMD ["python", "classifier_server.py"]