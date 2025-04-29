
# Failure‑Case Analysis

| Scenario                   | Misclassification                       | Caveat                                             |
|----------------------------|-----------------------------------------|----------------------------------------------------|
| Sarcastic posts            | Classified as neutral when negative     | Flag low-confidence cases for manual review        |
| Blurred images             | RandomForest outputs incorrect label    | Require manual inspection or higher-res images     |
| Geolocation outliers       | Rare lat/lon values confuse metadata    | Cross-validate with metadata extraction logs       |
