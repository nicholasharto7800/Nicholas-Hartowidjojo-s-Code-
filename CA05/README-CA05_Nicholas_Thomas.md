# CA05 – kNN-Based Movie Recommender

**Author:** Nicholas Thomas

## Overview

This project builds a content-based movie recommender system using the **k-Nearest Neighbors (kNN)** algorithm. Given a query movie — *The Post* — the system identifies the 5 most similar movies from a dataset based on genre and rating features.

## How It Works

Each movie is represented as a feature vector combining its IMDb rating and binary genre indicators. The kNN model measures similarity between movies using **Euclidean distance**, and returns the closest matches.

### Features Used

| Feature       | Type    | Description                          |
|---------------|---------|--------------------------------------|
| IMDB Rating   | Float   | Numeric rating from IMDb             |
| Biography     | Binary  | 1 if the movie is a biography        |
| Drama         | Binary  | 1 if the movie is a drama            |
| Thriller      | Binary  | 1 if the movie is a thriller         |
| Comedy        | Binary  | 1 if the movie is a comedy           |
| Crime         | Binary  | 1 if the movie is a crime film       |
| Mystery       | Binary  | 1 if the movie is a mystery          |
| History       | Binary  | 1 if the movie is a historical film  |

## Query Movie: *The Post*

The feature vector used for *The Post*:

| IMDB Rating | Biography | Drama | Thriller | Comedy | Crime | Mystery | History |
|-------------|-----------|-------|----------|--------|-------|---------|---------|
| 7.2         | 1         | 1     | 0        | 0      | 0     | 0       | 1       |

## Results

The top 5 movies most similar to *The Post*:

1. 12 Years a Slave
2. Hacksaw Ridge
3. Queen of Katwe
4. The Wind Rises
5. A Beautiful Mind

## Dependencies

- Python 3
- `pandas`
- `scikit-learn`

Install dependencies with:

```bash
pip install pandas scikit-learn
```

## Dataset

The movie dataset is loaded from the MSBA course data repository:

```
https://raw.githubusercontent.com/ArinB/MSBA-CA-Data/main/CA05/movies_recommendation_data.csv
```

## Usage

1. Clone or download the notebook `Nicholas_Thomas_CA05.ipynb`.
2. Ensure `movies_recommendation_data.csv` is in the same directory (or update the file path in the notebook).
3. Run all cells in order to train the kNN model and generate recommendations.

## Algorithm Details

- **Algorithm:** k-Nearest Neighbors (`sklearn.neighbors.NearestNeighbors`)
- **k:** 5 neighbors
- **Distance metric:** Euclidean distance
