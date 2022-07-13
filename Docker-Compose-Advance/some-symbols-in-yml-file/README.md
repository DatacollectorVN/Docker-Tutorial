# Docker-Tutorial

## Some Symbols In YML File
This example show some crucial symbol in YML file that are usually applied in docker-compose.yml file.

### Ampersand (&), Star (*) and Double arrow (<<)
In YAML, you can define anchors and later use them. For example:
```bash
foo: &myanchor
  key1: "val1"
  key2: "val2"

bar: *myanchor
```

In this code snippet, `&`defines an anchor names it `myanchor`, and `*myanchor` references that anchor. Now both foo and bar have the same keys and values.

Test to see output:
```bash
python load_yml_file --ymlfile example1.yml
```

The double arrow (`<<`)
```bash
foo: &myanchor
  key1: "val1"
  key2: "val2"

bar:
  << : *myanchor
  key2: "val2-new"
  key3: "val3"
```
`<<` is called the YAML merge key. You can compare it to class inheritance in OOP

Test to see output:
```bash
python load_yml_file --ymlfile example2.yml
```

### How it apply in docker-compose.yml file
```bash
x-common:
  &common
  image: apache/airflow:2.3.0
  user: "${AIRFLOW_UID}:0"
  env_file: 
    - .env
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - /var/run/docker.sock:/var/run/docker.sock

x-depends-on:
  &depends-on
  depends_on:
    postgres:
      condition: service_healthy
    airflow-init:
      condition: service_completed_successfully


webserver:
    <<: *common
    <<: *depends-on
    container_name: airflow-webserver
    restart: always
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 30s
      retries: 5
```

Test to see output:
```bash
python load_yml_file --ymlfile example3.yml
```