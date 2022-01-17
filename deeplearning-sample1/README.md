# Docker-Tutorial - Deeplearnining-Sample-1

This repository from [click-here](https://github.com/abhishekkrthakur/bert-sentiment). 

I follow the tutorial from [here](https://www.youtube.com/watch?v=0qG_0CPQhpg) for learning how to create Dockerfile for data science project.

## Experiment-1
**Searching Docker hub:**

![plot](src-imgs/figure_1.png)

**Dockerfile:**

![plot](src-imgs/figure_2.png)

**Build Docker image:**
```bash
docker build -f Dockerfile -t docker_tutorial .
```

**Run and access inside Docker Container:**
```bash
docker run -it docker_tutorial /bin/bash
```

*/bin/bash* for running bash inside container.

**Run htop inside container:** 
```bash
htop
```
*If it's works, mean you create successfully.*

## Experiment-2: Create python environment.

**Dockerfile:**

![plot](src-imgs/figure_3.png)

**Build Docker image:**
```bash
docker build -f Dockerfile -t docker_tutorial .
```

**Run and access inside Docker Container:**
```bash
docker run -it docker_tutorial /bin/bash
```

**Check python version inside container:**
```bash
python3
```