# Docker-Tutorial

## Docker-Compose
```bash
version: "3.9"
services:
  web:
    build: .
    ports:
      - "8000:5000"
    volumes:
      - .:/code
    environment:
      FLASK_ENV: development
  redis:
    image: "redis:alpine"
```

### Explaination:
- `version`: Version to use Docker Compose (recommend version 3).
- `services`: List all services that we use. In this case, we have 2 services `web` and `redis`.
- `build`: specifies the directory that contains the Dockerfile containing the instructions for building this service.
- `ports`: mapping of ports from container to host so that this port can be exposed to the outside world for accessing the app URL. Syntax `local_port`:`container_port`.
- `volumnes`: Use volume to mount a directory in the host with the container.
- `environment`: add environment variables. The specified variable is required for this image, and as its name suggests, configures the password for the root user of MySQL in this container.
- `image`: in the `redis` service, we pull image from Docker Hub.