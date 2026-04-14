# Serving

## Local

```bash
uv run uvicorn chest_xray_classifier.serving.main:app --host 0.0.0.0 --port 8000
```

Set `CHEST_XRAY_CLASSIFIER_CHECKPOINT=artifacts/checkpoints/best.ckpt` before launching so the app loads your trained weights.

## Docker

```bash
docker compose up api
```

## Endpoints

### `GET /health`

```json
{"status": "ok"}
```

### `POST /predict`

Multipart upload of one image:

```bash
curl -X POST -F "file=@test.jpeg" http://localhost:8000/predict
```

Response:

```json
{
  "pred": 1,
  "label": "normal",
  "probs": [0.02, 0.95, 0.03]
}
```

Class index order: `("bacterial_pneumonia", "normal", "viral_pneumonia")`.

### `GET /metrics`

Prometheus metrics including request count, latency histograms, and in-flight counts. Wire to Grafana via the standard scrape endpoint.

## Headers

Every response carries `X-Request-ID` for log correlation — propagate it from your upstream gateway to make traces end-to-end.
