#!/usr/bin/env bash

curl -X 'POST' \
  'http://localhost:8001/fit' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": {},
  "config": {
    "t": 0,
    "N": 24,
    "d": 0,
    "taup": 1,
    "taug": 1,
    "B": 0.05,
    "Cm": 0,
    "toff": 0,
    "tM": [
      7,
      11,
      17.5
    ],
    "seed": 0,
    "eps": 1e-18,
    "lb": [
      -12,
      0.5,
      0.1,
      0,
      0,
      -3
    ],
    "ub": [
      14,
      3,
      3,
      1,
      2,
      3
    ],
    "bounded_param_keys": [
      "d",
      "taup",
      "taug",
      "B",
      "Cm",
      "toff"
    ],
    "midbound": [
      0,
      1,
      1,
      0.05,
      0,
      0
    ],
    "bounded_param_descriptions": [
      "Time zone offset (hours)",
      "Photoperiod length",
      "Glucose response time constant",
      "Glucose Bias constant",
      "Cortisol temporal sensitivity coefficient",
      "Solar noon offset (latitude)"
    ],
    "bounds": {
      "lb": [
        0
      ],
      "ub": [
        0
      ],
      "keep_feasible": [
        true
      ]
    }
  }
}'