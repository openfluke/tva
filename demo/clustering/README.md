# K-Means Clustering Demo

This demo showcases LOOM's K-Means clustering capabilities with visual, colorized terminal output.

## Features

- **Demo 1**: Simple 2D point clustering with ASCII grid visualization
- **Demo 2**: Customer segmentation example with 3-feature analysis
- **Demo 3**: Finding optimal K using silhouette score analysis
- **Demo 4**: High-dimensional clustering (10D feature space)
- **Demo 5**: Performance comparison (parallel vs sequential execution)

## Running the Demo

```bash
cd tva/demo/clustering
go run main.go
```

## What You'll See

- Colored terminal output showing different clusters
- ASCII art visualization of 2D data
- Silhouette scores to measure clustering quality
- Performance metrics for parallel vs sequential execution
- Real-world example of customer segmentation

## How It Works

The demo uses LOOM's `nn.KMeansCluster()` function to automatically group data into clusters:

```go
centroids, assignments := nn.KMeansCluster(data, k, maxIter, parallel)
```

- **`data`**: Your feature vectors (e.g., `[][]float32`)
- **`k`**: Number of clusters to create
- **`maxIter`**: Maximum iterations (typically 100)
- **`parallel`**: Use parallel processing for large datasets

The function returns:
- **`centroids`**: The center point of each cluster
- **`assignments`**: Which cluster each data point belongs to

## Practical Applications

This clustering technique can be used for:
- Customer segmentation
- Image compression
- Anomaly detection
- Feature learning
- Document clustering
- Pattern recognition
