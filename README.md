# 🧠 AI Labs: A Collection of Artificial Intelligence Assignments

**AI Labs** is a repository containing a collection of laboratory assignments focused on fundamental Artificial Intelligence concepts. The project is primarily implemented in Python and covers various aspects of AI, including data clustering, genetic algorithms, and neural networks. It is designed for students and researchers looking to explore and apply key machine learning and artificial intelligence algorithms.

## 📝 Table of Contents

*   [✨ Features](#-features)
*   [🛠️ Technologies Used](#️-technologies-used)
*   [🚀 Quick Start](#-quick-start)
    *   [1️⃣ Clone the Repository](#1️⃣-clone-the-repository)
    *   [2️⃣ Install Dependencies](#2️⃣-install-dependencies)
    *   [3️⃣ Run the Labs](#3️⃣-run-the-labs)
*   [🔬 Lab Details](#-lab-details)
    *   [Lab1: Data Clustering](#lab1-data-clustering)
    *   [Lab2: Genetic Algorithms](#lab2-genetic-algorithms)
    *   [Lab3: Neural Networks (MNIST Classification)](#lab3-neural-networks-mnist-classification)
*   [🤝 Contributing](#-contributing)
*   [📄 License](#-license)
*   [🧰 Maintainer](#-maintainer)

## ✨ Features

*   **Comprehensive AI Study**: Covers key AI areas such as clustering, optimization, and deep learning.
*   **Practical Implementation**: Provides ready-to-run examples for each lab, demonstrating algorithm application.
*   **Python-Oriented Approach**: All assignments are implemented in Python, ensuring ease of understanding and code modification.
*   **Modular Structure**: Each lab is organized in a separate directory, simplifying navigation and study.
*   **Result Visualization**: Includes visualization examples for better understanding of clustering algorithms and neural networks.
*   **Popular Library Usage**: Employs standard and widely used Python libraries such as `scikit-learn`, `DEAP`, `Keras`, and `pandas`.

## 🛠️ Technologies Used

The project is built upon the following technologies and libraries:

*   **Python**: Primary programming language (version 3.x).
*   **pandas**: For data handling and analysis.
*   **scikit-learn**: For machine learning algorithms like K-Means and Mean Shift.
*   **matplotlib**: For data visualization.
*   **DEAP**: Framework for implementing genetic algorithms.
*   **numpy**: For numerical operations.
*   **Keras** (with TensorFlow backend): For building and training neural networks.

## 🚀 Quick Start

Follow these instructions to get the labs up and running:

### 1️⃣ Clone the Repository

```shell
git clone https://github.com/ArtemRivnyi/AI_labs.git
cd AI_labs
```

### 2️⃣ Install Dependencies

It is recommended to use a virtual environment. Install dependencies for each lab separately:

**For Lab1 (Data Clustering):**

```shell
pip install pandas scikit-learn matplotlib
```

**For Lab2 (Genetic Algorithms):**

```shell
pip install deap numpy
```

**For Lab3 (Neural Networks):**

```shell
pip install tensorflow keras numpy matplotlib
```

### 3️⃣ Run the Labs

Navigate to the respective lab directory and run the main script:

**Run Lab1:**

```shell
cd lab1
python main.py
```

**Run Lab2:**

```shell
cd lab2
python main.py
```

**Run Lab3:**

```shell
cd lab3
python main.py
```

## 🔬 Lab Details

### Lab1: Data Clustering

This lab focuses on data clustering techniques using `K-Means` and `Mean Shift` algorithms. It demonstrates how to load data, determine optimal cluster numbers using the silhouette score, and visualize the clustering results. The project uses `pandas` for data handling and `scikit-learn` for clustering algorithms.

**Key Concepts:**

*   K-Means Clustering
*   Mean Shift Clustering
*   Silhouette Score for optimal cluster determination
*   Data visualization with `matplotlib`

**Key Files:**

*   `main.py`: Python script implementing the clustering logic and visualization.
*   `lab01.csv`: Sample dataset used for clustering.
*   `lab01.ods`: OpenDocument Spreadsheet version of the sample dataset.

### Lab2: Genetic Algorithms

This lab implements a basic genetic algorithm to find the optimal solution for a given objective function. It utilizes the `DEAP` (Distributed Evolutionary Algorithms in Python) framework to define individuals, populations, fitness functions, and genetic operators (crossover, mutation, selection).

**Key Concepts:**

*   Genetic Algorithms fundamentals
*   Fitness function definition
*   Selection, Crossover, and Mutation operators
*   `DEAP` framework usage

**Key Files:**

*   `main.py`: Python script containing the genetic algorithm implementation.

### Lab3: Neural Networks (MNIST Classification)

This lab demonstrates the implementation of a simple neural network using `Keras` for classifying handwritten digits from the MNIST dataset. It covers data loading, preprocessing (normalization, one-hot encoding), model creation, training, evaluation, and saving the trained model.

**Key Concepts:**

*   Neural Network architecture (Sequential model)
*   Dense layers and activation functions (`relu`, `softmax`)
*   Categorical Cross-entropy loss and Adam optimizer
*   MNIST dataset classification
*   Model training and evaluation
*   Model persistence (`.keras` format)

**Key Files:**

*   `main.py`: Python script for building, training, and evaluating the neural network.
*   `mnist_model.h5`: A pre-trained Keras model (HDF5 format).
*   `my_model.keras`: The trained Keras model saved after execution.

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🧰 Maintainer

**Artem Rivnyi** — Junior Technical Support / DevOps Enthusiast

* 📧 [artemrivnyi@outlook.com](mailto:artemrivnyi@outlook.com)  
* 🔗 [LinkedIn](https://www.linkedin.com/in/artem-rivnyi/)  
* 🌐 [Personal Projects](https://personal-page-devops.onrender.com/)  
* 💻 [GitHub](https://github.com/ArtemRivnyi)
