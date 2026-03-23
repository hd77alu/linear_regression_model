# East Africa CO₂ Emission Predictor — Flutter App

## Description

A Flutter mobile application that predicts total CO₂ emissions (excluding LUCF) for East African countries using a trained linear regression model served via a FastAPI backend. Users can enter socioeconomic and sectoral emission indicators to get an instant prediction, either for a single entry or multiple entries at once.

The app connects to the deployed API at:
```
https://linear-regression-model-wk1e.onrender.com
```

---

## Project Structure

```text
east_africa_co2_prediction_mobile_app/
├── lib/
│   └── main.dart               # App entry point, theme, and full UI logic
├── android/                    # Android platform files
├── ios/                        # iOS platform files
├── web/                        # Web platform files
├── windows/                    # Windows platform files
├── linux/                      # Linux platform files
├── macos/                      # macOS platform files
├── pubspec.yaml                # Dependencies and project metadata
└── README.md
```

---

## Features

- **Single Prediction** — fill in 7 input fields and get one CO₂ emission prediction instantly
- **Batch Prediction** — switch to Multiple Entries mode, add as many rows as needed, and predict all at once
- **Expandable Results** — batch predictions are displayed as expandable cards (expanded by default), one per entry
- **Input Validation** — all fields are validated before submission; numeric fields reject negative values and non-numeric input
- **Error Handling** — API errors and network failures are displayed in a dedicated error box
- **Loading State** — the Predict button shows a spinner and is disabled while a request is in flight
- **Dark Theme** — dark blue (`#0A1E3C`) background with yellow (`#FFC107`) accents

---

## Input Fields

Each prediction requires exactly 7 inputs matching the model's training features:

| Field | Type | Example |
|---|---|---|
| Country | Text | Kenya |
| Year | Integer | 2020 |
| Population | Number | 53771300 |
| Transportation (Mt) | Number | 5.1 |
| Manufacturing / Construction (Mt) | Number | 2.3 |
| Electricity / Heat (Mt) | Number | 3.8 |
| Building (Mt) | Number | 1.7 |

---

## API Endpoints Used

| Mode | Method | Endpoint |
|---|---|---|
| Single | POST | `/predict` |
| Batch | POST | `/predict/batch` |

**Single request body:**
```json
{
  "country": "Kenya",
  "year": 2020,
  "population": 53771300,
  "transportation_mt": 5.1,
  "manufacturing_construction_mt": 2.3,
  "electricity_heat_mt": 3.8,
  "building_mt": 1.7
}
```

**Batch request body:**
```json
{
  "rows": [
    { "country": "Kenya", "year": 2020, "population": 53771300, "transportation_mt": 5.1, "manufacturing_construction_mt": 2.3, "electricity_heat_mt": 3.8, "building_mt": 1.7 },
    { "country": "Uganda", "year": 2019, "population": 45741000, "transportation_mt": 3.2, "manufacturing_construction_mt": 1.1, "electricity_heat_mt": 1.5, "building_mt": 0.9 }
  ]
}
```

---

## Setup Instructions

### Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) >= 3.18.0
- Dart >= 3.10.7
- A connected device or emulator (Android, iOS, or desktop)

### 1. Navigate to the app directory

```bash
cd summative/FlutterApp/east_africa_co2_prediction_mobile_app
```

### 2. Install dependencies

```bash
flutter pub get
```

### 3. Run the app

```bash
# Android emulator or connected device
flutter run -d android

# iOS simulator (macOS only)
flutter run -d ios

# Desktop (Windows)
flutter run -d windows

# Web (note: use port 3000 to avoid CORS issues with the API)
flutter run -d chrome --web-port 3000
```

### 4. Build for release

```bash
# Android APK
flutter build apk --release

# iOS (macOS only)
flutter build ios --release

# Web
flutter build web
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `http` | ^1.6.0 | HTTP requests to the prediction API |

---

## Notes

- The API is hosted on Render's free tier and may take 30–60 seconds to respond on the first request (cold start). The app has a 30-second timeout to handle this.
- When running on Flutter Web, ensure you use `--web-port 3000` due to CORS restrictions.
- Running on Android or iOS emulators does not have CORS restrictions.
