# API Reference

## Overview

The Culinary Agent API provides programmatic access to intelligent meal planning capabilities. Built with FastAPI, it offers automatic OpenAPI documentation and type-safe request/response handling.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement API keys or OAuth2.

## Endpoints

### Root

**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Culinary Agent API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "generate_plan": "/generate-plan",
    "docs": "/docs"
  }
}
```

---

### Health Check

**GET** `/health`

Check if the server and agent are running properly.

**Response:**
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-14B-Instruct",
  "version": "1.0.0"
}
```

**Status Values:**
- `healthy`: Agent fully loaded and ready
- `initializing`: Server starting up
- `error`: Initialization failed

---

### Generate Meal Plan

**POST** `/generate-plan`

Generate or modify meal plans based on dietary constraints.

**Request Body:**

```json
{
  "diet_constraints": "string",
  "mode": "Full Day" | "Breakfast" | "Lunch" | "Dinner",
  "previous_plan": "string | null"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `diet_constraints` | string | Yes | Dietary preferences (e.g., "vegan", "high protein") |
| `mode` | string | Yes | What to generate/update |
| `previous_plan` | string | No | Previous plan text for updates |

**Mode Options:**
- `Full Day`: Generate complete meal plan (Breakfast, Lunch, Dinner)
- `Breakfast`: Update only breakfast
- `Lunch`: Update only lunch
- `Dinner`: Update only dinner

**Response:**

```json
{
  "status": "success",
  "result": "string",
  "mode": "string",
  "diet_constraints": "string"
}
```

**Error Responses:**

- `503 Service Unavailable`: Agent not initialized
- `500 Internal Server Error`: Plan generation failed

**Example Request:**

```bash
curl -X POST "http://localhost:8000/generate-plan" \
  -H "Content-Type: application/json" \
  -d '{
    "diet_constraints": "vegan, gluten-free",
    "mode": "Full Day",
    "previous_plan": null
  }'
```

**Example Response:**

```json
{
  "status": "success",
  "result": "==================== BREAKFAST ====================\n...",
  "mode": "Full Day",
  "diet_constraints": "vegan, gluten-free"
}
```

---

## Interactive Documentation

Access the interactive Swagger UI at:
```
http://localhost:8000/docs
```

Access the ReDoc documentation at:
```
http://localhost:8000/redoc
```

## Rate Limiting

Currently no rate limiting is implemented. For production, consider adding rate limiting middleware.

## CORS

The API allows all origins by default. For production, configure specific allowed origins in `src/api/server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

All endpoints return JSON responses with consistent error structure:

```json
{
  "detail": "Error message"
}
```

## Usage Examples

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/generate-plan",
    json={
        "diet_constraints": "keto",
        "mode": "Full Day"
    }
)

plan = response.json()["result"]
print(plan)
```

### JavaScript

```javascript
fetch('http://localhost:8000/generate-plan', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    diet_constraints: 'paleo',
    mode: 'Full Day'
  })
})
.then(res => res.json())
.then(data => console.log(data.result));
```

## Versioning

Current version: 1.0.0

The API follows semantic versioning. Breaking changes will increment the major version.
