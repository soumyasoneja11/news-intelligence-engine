# News Intelligence Engine — production image
FROM python:3.11-slim

WORKDIR /app

# faiss-cpu and scikit-learn need OpenMP at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the search index at image build time so containers start quickly.
# Set BUILD_INDEX=false to skip (entrypoint will build on first boot instead).
ARG BUILD_INDEX=true
RUN if [ "$BUILD_INDEX" = "true" ]; then python src/pipeline.py; fi

ENV PORT=8501
EXPOSE 8501

CMD ["python", "scripts/entrypoint.py"]
