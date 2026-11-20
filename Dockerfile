# Same environment that produced outputs/.
#   docker build -t incognito-t4 .
#   docker run --rm -v "$PWD/videos:/videos" incognito-t4 /videos/P01_11.mp4 /videos/P01_11.json
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.sh ./
COPY code ./code
COPY weights ./weights
RUN chmod +x run.sh
ENV TEAM_ID=INCOGNITO
ENTRYPOINT ["./run.sh"]
