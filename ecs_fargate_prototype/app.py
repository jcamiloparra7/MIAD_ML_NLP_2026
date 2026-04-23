from datetime import datetime, timezone
import os

from flask import Flask, render_template_string
from flask_restx import Api, Resource, fields

from inference import predict_proba

app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="Phishing Prediction API",
    description="Prototype API for phishing URL inference with scikit-learn.",
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

predict_request = api.model(
    "PredictRequest",
    {
        "url": fields.String(required=True, description="URL to analyze"),
    },
)

predict_response = api.model(
    "PredictResponse",
    {
        "url": fields.String(description="Normalized URL used for inference"),
        "phishing_probability": fields.Float(description="Predicted phishing probability"),
    },
)


@api.route("/health")
class Health(Resource):
    @api.marshal_with(health_response)
    def get(self):
        return {
            "status": "ok",
            "service": "phishing-api",
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
            <title>Phishing API Status</title>
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
              <h1>Phishing Prediction API</h1>
              <p>The service is up and ready to score phishing URLs.</p>
              <section class="meta">
                <div class="meta-row"><strong>Service</strong><span>phishing-api</span></div>
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
        url = api.payload["url"].strip()
        if not url:
            api.abort(400, "'url' must be a non-empty string.")

        return {
            "url": url,
            "phishing_probability": predict_proba(url),
        }, 200


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
    )
