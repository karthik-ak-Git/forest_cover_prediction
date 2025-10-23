# 📚 API Documentation - Forest Cover Prediction

Complete API reference for the Forest Cover Type Prediction service.

## Base URL

- **Development**: `http://localhost:8000`
- **Staging**: `https://staging-api.forest-cover.example.com`
- **Production**: `https://api.forest-cover.example.com`

## Authentication

Most endpoints require authentication using JWT tokens.

### Get Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Using Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Endpoints

### 1. Health Check

Check API health status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-10-23T10:00:00Z",
  "checks": {
    "database": "ok",
    "redis": "ok",
    "model": "loaded"
  }
}
```

---

### 2. Single Prediction

Make a prediction for a single sample.

```http
POST /predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "elevation": 2596,
  "aspect": 51,
  "slope": 3,
  "horizontal_distance_to_hydrology": 258,
  "vertical_distance_to_hydrology": 0,
  "horizontal_distance_to_roadways": 510,
  "hillshade_9am": 221,
  "hillshade_noon": 232,
  "hillshade_3pm": 148,
  "horizontal_distance_to_fire_points": 6279,
  "wilderness_area_0": 1,
  "wilderness_area_1": 0,
  "wilderness_area_2": 0,
  "wilderness_area_3": 0,
  "soil_type_0": 0,
  "soil_type_1": 0,
  ...
  "soil_type_39": 0
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Spruce/Fir",
  "confidence": 0.95,
  "probabilities": {
    "1": 0.95,
    "2": 0.02,
    "3": 0.01,
    "4": 0.01,
    "5": 0.00,
    "6": 0.01,
    "7": 0.00
  },
  "processing_time_ms": 12.5,
  "model_version": "v1.2.0",
  "timestamp": "2025-10-23T10:00:00Z"
}
```

---

### 3. Batch Prediction

Make predictions for multiple samples.

```http
POST /predict/batch
Content-Type: application/json
Authorization: Bearer <token>

{
  "samples": [
    {
      "elevation": 2596,
      "aspect": 51,
      ...
    },
    {
      "elevation": 2590,
      "aspect": 56,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "index": 0,
      "prediction": 1,
      "prediction_label": "Spruce/Fir",
      "confidence": 0.95
    },
    {
      "index": 1,
      "prediction": 2,
      "prediction_label": "Lodgepole Pine",
      "confidence": 0.92
    }
  ],
  "total_samples": 2,
  "processing_time_ms": 25.3,
  "timestamp": "2025-10-23T10:00:00Z"
}
```

---

### 4. Prediction with Explainability (SHAP)

Get prediction with SHAP explanation.

```http
POST /predict/explain
Content-Type: application/json
Authorization: Bearer <token>

{
  "elevation": 2596,
  "aspect": 51,
  "slope": 3,
  ...
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Spruce/Fir",
  "confidence": 0.95,
  "explanation": {
    "shap_values": {
      "elevation": 0.45,
      "slope": 0.23,
      "aspect": -0.12,
      "horizontal_distance_to_hydrology": 0.18,
      ...
    },
    "base_value": 0.14,
    "top_features": [
      {
        "feature": "elevation",
        "value": 2596,
        "contribution": 0.45,
        "impact": "positive"
      },
      {
        "feature": "slope",
        "value": 3,
        "contribution": 0.23,
        "impact": "positive"
      }
    ]
  },
  "processing_time_ms": 85.2
}
```

---

### 5. Model Information

Get model metadata and performance metrics.

```http
GET /model/info
Authorization: Bearer <token>
```

**Response:**
```json
{
  "model_name": "RandomForestClassifier",
  "version": "v1.2.0",
  "trained_date": "2025-10-15T14:30:00Z",
  "metrics": {
    "accuracy": 0.974,
    "precision": 0.973,
    "recall": 0.972,
    "f1_score": 0.972
  },
  "classes": [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz"
  ],
  "features": {
    "total": 54,
    "numerical": 10,
    "categorical": 44
  },
  "training_samples": 15120
}
```

---

### 6. Feature Importance

Get model feature importance rankings.

```http
GET /model/feature-importance
Authorization: Bearer <token>
```

**Response:**
```json
{
  "feature_importances": [
    {
      "feature": "elevation",
      "importance": 0.28,
      "rank": 1
    },
    {
      "feature": "horizontal_distance_to_roadways",
      "importance": 0.15,
      "rank": 2
    },
    {
      "feature": "wilderness_area_3",
      "importance": 0.12,
      "rank": 3
    }
  ],
  "top_10": [...]
}
```

---

### 7. Model Performance by Class

Get detailed performance metrics for each class.

```http
GET /model/performance
Authorization: Bearer <token>
```

**Response:**
```json
{
  "overall": {
    "accuracy": 0.974,
    "macro_avg_f1": 0.968
  },
  "per_class": {
    "1": {
      "name": "Spruce/Fir",
      "precision": 0.98,
      "recall": 0.97,
      "f1_score": 0.975,
      "support": 2160
    },
    "2": {
      "name": "Lodgepole Pine",
      "precision": 0.96,
      "recall": 0.98,
      "f1_score": 0.970,
      "support": 2456
    }
  }
}
```

---

### 8. Prediction History

Get prediction history for the authenticated user.

```http
GET /predictions/history?limit=10&offset=0
Authorization: Bearer <token>
```

**Response:**
```json
{
  "predictions": [
    {
      "id": "pred_12345",
      "timestamp": "2025-10-23T10:00:00Z",
      "prediction": 1,
      "confidence": 0.95,
      "input_hash": "a1b2c3d4"
    }
  ],
  "total": 150,
  "limit": 10,
  "offset": 0
}
```

---

### 9. Validate Input Data

Validate input data without making a prediction.

```http
POST /validate
Content-Type: application/json
Authorization: Bearer <token>

{
  "elevation": 2596,
  "aspect": 51,
  ...
}
```

**Response:**
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [
    "Elevation is in the upper 10% of training data"
  ],
  "data_quality_score": 0.98
}
```

---

### 10. Metrics (Prometheus)

Get Prometheus metrics for monitoring.

```http
GET /metrics
```

**Response:** (Prometheus format)
```
# HELP prediction_requests_total Total prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total 1523

# HELP prediction_latency_seconds Prediction latency
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.01"} 1234
prediction_latency_seconds_bucket{le="0.05"} 1450
...
```

---

### 11. Model Drift Detection

Check for model drift.

```http
GET /model/drift?window=7d
Authorization: Bearer <token>
```

**Response:**
```json
{
  "drift_detected": false,
  "drift_score": 0.02,
  "threshold": 0.05,
  "period": "7 days",
  "metrics": {
    "feature_drift": {
      "elevation": 0.01,
      "aspect": 0.03
    },
    "prediction_drift": 0.02
  },
  "recommendation": "No action needed"
}
```

---

### 12. Retrain Model

Trigger model retraining (admin only).

```http
POST /model/retrain
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "data_source": "s3://bucket/new-training-data.csv",
  "config": {
    "n_estimators": 200,
    "max_depth": 20
  }
}
```

**Response:**
```json
{
  "job_id": "retrain_789",
  "status": "started",
  "estimated_duration_minutes": 30,
  "callback_url": "/model/retrain/status/retrain_789"
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "error": "Validation Error",
  "detail": [
    {
      "loc": ["body", "elevation"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 401 Unauthorized

```json
{
  "error": "Unauthorized",
  "detail": "Invalid or expired token"
}
```

### 429 Too Many Requests

```json
{
  "error": "Rate Limit Exceeded",
  "detail": "Maximum 100 requests per minute exceeded",
  "retry_after": 45
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal Server Error",
  "detail": "An unexpected error occurred",
  "request_id": "req_abc123"
}
```

---

## Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/predict` | 100 req/min | Per user |
| `/predict/batch` | 20 req/min | Per user |
| `/predict/explain` | 50 req/min | Per user |
| Other endpoints | 200 req/min | Per user |

---

## SDKs and Code Examples

### Python

```python
import requests

API_BASE_URL = "https://api.forest-cover.example.com"
API_TOKEN = "your_token_here"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Make prediction
data = {
    "elevation": 2596,
    "aspect": 51,
    "slope": 3,
    # ... other features
}

response = requests.post(
    f"{API_BASE_URL}/predict",
    headers=headers,
    json=data
)

result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_BASE_URL = 'https://api.forest-cover.example.com';
const API_TOKEN = 'your_token_here';

async function makePrediction(data) {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/predict`,
      data,
      {
        headers: {
          'Authorization': `Bearer ${API_TOKEN}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    console.log('Prediction:', response.data.prediction_label);
    console.log('Confidence:', response.data.confidence);
    
    return response.data;
  } catch (error) {
    console.error('Error:', error.response.data);
  }
}
```

### cURL

```bash
curl -X POST "https://api.forest-cover.example.com/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "elevation": 2596,
    "aspect": 51,
    "slope": 3
  }'
```

---

## WebSocket API (Real-time Predictions)

### Connect

```javascript
const ws = new WebSocket('wss://api.forest-cover.example.com/ws/predict');

ws.onopen = () => {
  // Send authentication
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'YOUR_TOKEN'
  }));
  
  // Send prediction request
  ws.send(JSON.stringify({
    type: 'predict',
    data: {
      elevation: 2596,
      aspect: 51,
      ...
    }
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Prediction:', result);
};
```

---

## Versioning

API versioning is done via URL path:

- **v1** (current): `/api/v1/...`
- **v2** (beta): `/api/v2/...`

---

## Support

- **Documentation**: https://docs.forest-cover.example.com
- **API Status**: https://status.forest-cover.example.com
- **GitHub Issues**: https://github.com/karthik-ak-Git/forest_cover_prediction/issues
- **Email**: api-support@example.com

---

**Last Updated**: October 2025
**API Version**: 1.0.0
