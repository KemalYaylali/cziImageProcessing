# Mathematical Background for CZI Image Processing

## 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)

CLAHE is an advanced form of histogram equalization used to enhance image contrast.

### Basic Histogram Equalization

For an image I with intensity levels in the range [0, L-1], the histogram equalization transform is:

h(v) = round((L-1) * cumsum(H(v)) / (image width * image height))

Where:
- v is the input intensity level
- H(v) is the histogram of the image
- cumsum is the cumulative sum

### CLAHE Modification

CLAHE divides the image into small tiles (e.g., 8x8 pixels). For each tile:

1. Compute the histogram.
2. Clip the histogram at a predefined value to limit contrast:
   
   H_clipped(v) = min(H(v), clip_limit)

3. Redistribute the clipped pixels equally among all histogram bins.
4. Compute the equalization transform for the tile.

The final pixel value is interpolated from the transforms of neighboring tile corners to eliminate artificially induced boundaries.

## 2. U-Net Architecture

U-Net is a convolutional neural network architecture designed for image segmentation.

### Key Components:

1. **Convolutional Layers**: Apply filters F to input X:
   
   (F * X)(i,j) = ∑_m ∑_n F(m,n) * X(i-m, j-n)

2. **Max Pooling**: Downsampling operation that reduces spatial dimensions:
   
   MaxPool(X)(i,j) = max(X(2i:2i+1, 2j:2j+1))

3. **Transposed Convolution**: Upsampling operation:
   
   (F *_t X)(i,j) = ∑_m ∑_n F(m,n) * X(i+m, j+n)

4. **Skip Connections**: Concatenate features from encoder to decoder.

5. **Activation Function (ReLU)**: 
   
   ReLU(x) = max(0, x)

### Loss Function

Binary Cross-Entropy Loss:

L = -1/N ∑_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]

Where:
- y_i is the true label (0 or 1)
- p_i is the predicted probability

## 3. Watershed Algorithm

The watershed algorithm treats the image as a topographic surface where pixel values represent heights.

1. **Distance Transform**: For a binary image B, the distance transform D is:
   
   D(p) = min{d(p,q) : B(q) = 0}

   Where d(p,q) is the Euclidean distance between pixels p and q.

2. **Marker Identification**: Find local maxima in the distance transform.

3. **Flooding Process**: Conceptually, water rises from markers. Watershed lines are formed where different catchment basins meet.

Mathematically, this can be formulated as a minimum spanning forest problem in the graph representation of the image.

## 4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN groups together points that are closely packed together in space.

Key Concepts:

1. **ε-neighborhood**: N_ε(p) = {q ∈ D | dist(p,q) ≤ ε}
   Where D is the dataset and dist is a distance function.

2. **Core Point**: A point p is a core point if |N_ε(p)| ≥ MinPts

3. **Directly Density-Reachable**: A point q is directly density-reachable from p if:
   - p is a core point
   - q ∈ N_ε(p)

4. **Density-Reachable**: A point q is density-reachable from p if there is a chain of points p1, ..., pn with p1 = p and pn = q where each pi+1 is directly density-reachable from pi.

5. **Density-Connected**: Two points p and q are density-connected if there exists a point o such that both p and q are density-reachable from o.

The DBSCAN algorithm forms clusters by connecting density-connected points.

## 5. Image Segmentation Evaluation Metrics

### Dice Coefficient

The Dice coefficient measures the overlap between two segmentations:

Dice = (2 * |X ∩ Y|) / (|X| + |Y|)

Where X and Y are the predicted and ground truth segmentations respectively.

### Intersection over Union (IoU)

IoU, also known as the Jaccard index, is another measure of overlap:

IoU = |X ∩ Y| / |X ∪ Y|

## 6. Morphological Operations

### Dilation

Dilation expands shapes in an image. For a binary image A and structuring element B:

A ⊕ B = {z | (B_z ∩ A) ≠ ∅}

Where B_z is the translation of B by vector z.

### Erosion

Erosion shrinks shapes in an image:

A ⊖ B = {z | B_z ⊆ A}

### Opening and Closing

Opening (erosion followed by dilation):

A ○ B = (A ⊖ B) ⊕ B

Closing (dilation followed by erosion):

A • B = (A ⊕ B) ⊖ B

These operations are used to remove small objects (opening) or fill small holes (closing).

Understanding these mathematical foundations provides insight into how each algorithm works and can guide parameter selection and algorithm modifications for specific use cases in CZI image processing.

