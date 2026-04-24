from datetime import datetime, timezone
import os

from flask import Flask, render_template_string
from flask_restx import Api, Resource, fields

from inference import predict_popularity

app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="Spotify Popularity Prediction API",
    description="Prototype API for batch Spotify popularity inference with scikit-learn.",
    doc="/",
)

health_response = api.model(
    "HealthResponse",
    {
        "status": fields.String(description="Service health status"),
        "service": fields.String(description="Service name"),
        "time": fields.String(description="Current UTC timestamp"),
        "docs": fields.String(description="Swagger UI path"),
        "status_page": fields.String(description="Browser-friendly status page path"),
    },
)

track_features = api.model(
    "TrackFeatures",
    {
        "track_name": fields.String(required=True, description="Track name"),
        "album_name": fields.String(required=True, description="Album name"),
        "artists": fields.String(required=True, description="Comma-separated artist names"),
        "track_genre": fields.String(required=True, description="Primary track genre"),
        "duration_ms": fields.Float(required=True, description="Track duration in milliseconds"),
        "explicit": fields.Boolean(required=True, description="Whether the track is explicit"),
        "danceability": fields.Float(required=True, description="Danceability score"),
        "energy": fields.Float(required=True, description="Energy score"),
        "key": fields.Integer(required=True, description="Detected key"),
        "loudness": fields.Float(required=True, description="Average loudness"),
        "mode": fields.Integer(required=True, description="Major/minor mode"),
        "speechiness": fields.Float(required=True, description="Speechiness score"),
        "acousticness": fields.Float(required=True, description="Acousticness score"),
        "instrumentalness": fields.Float(required=True, description="Instrumentalness score"),
        "liveness": fields.Float(required=True, description="Liveness score"),
        "valence": fields.Float(required=True, description="Valence score"),
        "tempo": fields.Float(required=True, description="Estimated tempo"),
        "time_signature": fields.Integer(required=True, description="Estimated time signature"),
    },
)

predict_request = api.model(
    "PredictRequest",
    {
        "instances": fields.List(
            fields.Nested(track_features),
            required=True,
            description="One or more Spotify track records to score",
        ),
    },
)

predict_response = api.model(
    "PredictResponse",
    {
        "predictions": fields.List(
            fields.Float(description="Predicted popularity score"),
            description="Predicted popularity score for each input record",
        ),
        "count": fields.Integer(description="Number of records scored"),
    },
)


@api.route("/health")
class Health(Resource):
    @api.marshal_with(health_response)
    def get(self):
        return {
            "status": "ok",
            "service": "spotify-popularity-api",
            "time": datetime.now(timezone.utc).isoformat(),
            "docs": "/",
            "status_page": "/status",
        }, 200


@app.get("/status")
def status_page():
    return render_template_string(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Spotify Popularity API Status</title>
            <style>
              :root {
                color-scheme: light dark;
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              }
              body {
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                background: linear-gradient(135deg, #0f172a, #1d4ed8);
              }
              .card {
                width: min(560px, calc(100vw - 2rem));
                padding: 2rem;
                border-radius: 20px;
                background: rgba(15, 23, 42, 0.88);
                color: #e2e8f0;
                box-shadow: 0 24px 60px rgba(15, 23, 42, 0.35);
              }
              .badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.35rem 0.85rem;
                border-radius: 999px;
                background: rgba(34, 197, 94, 0.18);
                color: #86efac;
                font-weight: 700;
              }
              .dot {
                width: 0.6rem;
                height: 0.6rem;
                border-radius: 50%;
                background: currentColor;
              }
              h1 {
                margin: 1rem 0 0.5rem;
                font-size: 2rem;
              }
              p {
                color: #cbd5e1;
                line-height: 1.6;
              }
              .meta {
                margin-top: 1.5rem;
                display: grid;
                gap: 0.75rem;
              }
              .meta-row {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                padding: 0.9rem 1rem;
                border-radius: 12px;
                background: rgba(148, 163, 184, 0.12);
              }
              a {
                color: #93c5fd;
                text-decoration: none;
              }
              a:hover {
                text-decoration: underline;
              }
            </style>
          </head>
          <body>
            <main class="card">
              <div class="badge"><span class="dot"></span>Healthy</div>
              <h1>Spotify Popularity Prediction API</h1>
              <p>The service is up and ready to score Spotify tracks.</p>
              <section class="meta">
                <div class="meta-row"><strong>Service</strong><span>spotify-popularity-api</span></div>
                <div class="meta-row"><strong>Time (UTC)</strong><span>{{ timestamp }}</span></div>
                <div class="meta-row"><strong>Swagger UI</strong><a href="/">Open docs</a></div>
                <div class="meta-row"><strong>JSON healthcheck</strong><a href="/health">/health</a></div>
              </section>
            </main>
          </body>
        </html>
        """,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@api.route("/predict")
class Predict(Resource):
    @api.expect(predict_request, validate=True)
    @api.marshal_with(predict_response)
    def post(self):
        instances = api.payload["instances"]
        if not instances:
            api.abort(400, "'instances' must contain at least one record.")

        try:
            predictions = predict_popularity(instances)
        except ValueError as error:
            api.abort(400, str(error))

        return {
            "predictions": predictions,
            "count": len(predictions),
        }, 200


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
    )
