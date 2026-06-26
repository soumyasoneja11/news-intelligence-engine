# News Intelligence Engine — production image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-build.txt

COPY . .

ARG BUILD_INDEX=true
ENV TRENDING_REFERENCE=auto
RUN if [ "$BUILD_INDEX" = "true" ]; then python src/pipeline.py; fi

ENV PORT=8501
ENV ALLOW_REBUILD=false
EXPOSE 8501

CMD ["python", "scripts/entrypoint.py"]
