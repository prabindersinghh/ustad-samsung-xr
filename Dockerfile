# Same environment that produced outputs/. Weights end up inside the image either way:
# copied from weights/ when present (the submitted TEAM_ID/ folder), otherwise
# downloaded and checksum-verified by get_weights.sh at build time.
#   docker build -t incognito-t4 .
#   docker run --rm -v "$PWD/videos:/videos" incognito-t4 /videos/P01_11.mp4 /videos/P01_11.json
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.sh get_weights.sh validate.py ./
COPY code ./code
COPY weights ./weights
RUN chmod +x run.sh get_weights.sh && ./get_weights.sh
ENV TEAM_ID=Thapar_INCOGNITO
ENTRYPOINT ["./run.sh"]
