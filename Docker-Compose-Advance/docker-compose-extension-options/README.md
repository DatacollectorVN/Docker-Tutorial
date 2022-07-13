# Docker-Tutorial

## Docker-compose extension (3.4+ version)

### 1. Extension fields 
If you can use 3.4+ compose files, [extension fields](https://github.com/compose-spec/compose-spec/blob/master/spec.md#extension) are probably the best option:
```bash
version: '3.4'

 x-common-variables: &common-variables
   VARIABLE: some_value
   ANOTHER_VARIABLE: another_value

 services:
   some_service:
     image: someimage
     environment: *common-variables

   another_service:
     image: anotherimage
     environment:
       <<: *common-variables
       NON_COMMON_VARIABLE: 'non_common_value'
```

### 2. The `env_file` configuration option
You can pass multiple environment variables from an external file through to a service’s containers with the [env_file](https://docs.docker.com/compose/environment-variables/#the-env_file-configuration-option) option.

`docker-compose.yml`
```bash
version: '3.2'

services:
  some_service:
    image: someimage
    env_file:
      - 'variables.env'

  another_service:
    image: anotherimage
    env_file:
      - 'variables.env'
```
`variables.env`
```bash
VARIABLE=some_value
ANOTHER_VARIABLE=another_value
```
### 3. The `.env` file 
The [env file](https://docs.docker.com/compose/environment-variables/#the-env-file) in project root (or variables at actual compose environment).

`docker-compose.yml`
```bash
version: '3.2'

services:
  some_service:
    image: someimage
    environment:
      - VARIABLE

  another_service:
    image: anotherimage
    environment:
      - VARIABLE
      - ANOTHER_VARIABLE
```

`.env`
```bash
VARIABLE=some_value
ANOTHER_VARIABLE=another_value
```