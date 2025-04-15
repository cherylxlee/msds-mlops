FROM python:3.9-slim
RUN pip install mlflow
EXPOSE 5000
WORKDIR /mlflow
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
