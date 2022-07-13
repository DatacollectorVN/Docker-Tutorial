# Docker-Tutorial

## Docker-Compose
```bash
version: "3"
services:
  app:
    build: ./app
    links:
      - db
    ports:
      - "5001:5000"

  db:
    image: mysql:latest
    command: --default-authentication-plugin=caching_sha2_password
    ports:
      - "32000:3306"
    environment:
      MYSQL_ROOT_PASSWORD: root
    volumes:
      - ./db:/docker-entrypoint-initdb.d/:ro
```

### Explaination:
- `build`: specifies the directory that contains the Dockerfile containing the instructions for building this service.
- `links`: links this service to another container. This will also allow you to use the name of the service instead of having to find the ip of the database container, and express a dependency which will determine the order of start up of the container.
- `ports`: mapping of ports from container to host so that this port can be exposed to the outside world for accessing the app URL.
- `image`: similar to the FROM instruction in the Dockerfile. Instead of writing a new Dockerfile, I am using an existing image from a repository. It’s important to specify the version. If your installed mysql client is not of the same version problems may occur.