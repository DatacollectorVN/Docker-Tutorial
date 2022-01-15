# Docker-Tutorial - Web-app 

This tutorial used [FastAPI](https://fastapi.tiangolo.com/)

## FOR SETUP 
### 1. Setup enviroment

```bash
conda create -n web-app-docker python=3.8

conda activate web-app-docker
```

### 2. Install dependencies

```bash
pip install fastapi

pip install "uvicorn[standard]"
```

```bash 
pip freeze > requirements.txt
```

## FOR running Docker

### 1. Build Images from Dockerfile

```bash 
docker build -t python-fastapi .
```

### 2. Run Container

```bash
docker run -p 8000:8000 python-fastapi
```