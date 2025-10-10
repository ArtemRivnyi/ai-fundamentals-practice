# AI Labs

This repository contains a collection of laboratory assignments focused on Artificial Intelligence concepts, implemented primarily in Python. Each lab explores different aspects of AI, including clustering, genetic algorithms, and neural networks.

## Project Structure

The project is organized into several modules, each representing a distinct laboratory assignment:

```
AI_labs-main/
├── lab1/             # Data Clustering (K-Means, Mean Shift)
├── lab2/             # Genetic Algorithms
└── lab3/             # Neural Networks (MNIST Classification)
```

## Lab Details

### Lab1: Data Clustering

This lab focuses on data clustering techniques using `K-Means` and `Mean Shift` algorithms. It demonstrates how to load data, determine optimal cluster numbers using the silhouette score, and visualize the clustering results. The project uses `pandas` for data handling and `scikit-learn` for clustering algorithms.

**Key Concepts:**
-   K-Means Clustering
-   Mean Shift Clustering
-   Silhouette Score for optimal cluster determination
-   Data visualization with `matplotlib`

**Key Files:**
-   `main.py`: Python script implementing the clustering logic and visualization.
-   `lab01.csv`: Sample dataset used for clustering.
-   `lab01.ods`: OpenDocument Spreadsheet version of the sample dataset.

**Dependencies:**
-   `pandas`
-   `scikit-learn`
-   `matplotlib`

### Lab2: Genetic Algorithms

This lab implements a basic genetic algorithm to find the optimal solution for a given objective function. It utilizes the `DEAP` (Distributed Evolutionary Algorithms in Python) framework to define individuals, populations, fitness functions, and genetic operators (crossover, mutation, selection).

**Key Concepts:**
-   Genetic Algorithms fundamentals
-   Fitness function definition
-   Selection, Crossover, and Mutation operators
-   `DEAP` framework usage

**Key Files:**
-   `main.py`: Python script containing the genetic algorithm implementation.

**Dependencies:**
-   `deap`
-   `numpy`

### Lab3: Neural Networks (MNIST Classification)

This lab demonstrates the implementation of a simple neural network using `Keras` for classifying handwritten digits from the MNIST dataset. It covers data loading, preprocessing (normalization, one-hot encoding), model creation, training, evaluation, and saving the trained model.

**Key Concepts:**
-   Neural Network architecture (Sequential model)
-   Dense layers and activation functions (`relu`, `softmax`)
-   Categorical Cross-entropy loss and Adam optimizer
-   MNIST dataset classification
-   Model training and evaluation
-   Model persistence (`.keras` format)

**Key Files:**
-   `main.py`: Python script for building, training, and evaluating the neural network.
-   `mnist_model.h5`: A pre-trained Keras model (HDF5 format).
-   `my_model.keras`: The trained Keras model saved after execution.

**Dependencies:**
-   `keras` (with TensorFlow backend)
-   `numpy`
-   `matplotlib`

## Getting Started

To run these projects, you will need Python 3.x and the specified libraries installed. It is recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AI_labs-main.git
    cd AI_labs-main
    ```
    *(Note: Replace `https://github.com/your-username/AI_labs-main.git` with the actual repository URL if it's hosted on GitHub.)*

2.  **Install dependencies for each lab:**

    For `lab1`:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```

    For `lab2`:
    ```bash
    pip install deap numpy
    ```

    For `lab3`:
    ```bash
    pip install tensorflow keras numpy matplotlib
    ```

3.  **Navigate to a specific lab directory and run the main script:**

    For `lab1`:
    ```bash
    cd lab1
    python main.py
    ```

    For `lab2`:
    ```bash
    cd lab2
    python main.py
    ```

    For `lab3`:
    ```bash
    cd lab3
    python main.py
    ```

## Contributing

Contributions are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: A `LICENSE` file is not included in the provided archive, so this is a placeholder. Please create one if needed.)

---

**Author:** Artem Rivnyi

