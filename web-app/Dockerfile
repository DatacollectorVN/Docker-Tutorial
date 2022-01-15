FROM python:3.8

WORKDIR /fastapi-app

COPY requirements.txt .
COPY ./app ./app

RUN pip install -r requirements.txt

CMD ["python", "./app/main.py"]