# 🧠 AI Fundamentals Practice: A Collection of Artificial Intelligence Assignments

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI Fundamentals Practice** is a repository containing a collection of laboratory assignments focused on fundamental Artificial Intelligence concepts. The project is primarily implemented in Python and covers various aspects of AI, including data clustering, genetic algorithms, and neural networks. It is designed for students and researchers looking to explore and apply key machine learning and artificial intelligence algorithms.

## 📝 Table of Contents

* [✨ Features](#-features)
* [🛠️ Technologies Used](#️-technologies-used)
* [🚀 Quick Start](#-quick-start)
    * [1️⃣ Clone the Repository](#1️⃣-clone-the-repository)
    * [2️⃣ Install Dependencies](#2️⃣-install-dependencies)
    * [3️⃣ Run the Labs](#3️⃣-run-the-labs)
* [🔬 Lab Details](#-lab-details)
    * [Lab1: Data Clustering](#lab1-data-clustering)
        * [Screenshots](#lab1-screenshots)
        * [What I Learned](#lab1-what-i-learned)
        * [Use Cases](#lab1-use-cases)
    * [Lab2: Genetic Algorithms](#lab2-genetic-algorithms)
        * [Results](#lab2-results)
        * [What I Learned](#lab2-what-i-learned-1)
        * [Use Cases](#lab2-use-cases)
    * [Lab3: Neural Networks (MNIST Classification)](#lab3-neural-networks-mnist-classification)
        * [Screenshots](#lab3-screenshots)
        * [What I Learned](#lab3-what-i-learned-2)
        * [Use Cases](#lab3-use-cases)
* [🤝 Contributing](#-contributing)
* [📄 License](#-license)
* [🧰 Maintainer](#-maintainer)

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

#### 📸 Screenshots {#lab1-screenshots}

The analysis involved four key visualizations to understand the data and the clustering process: the original data distribution, the centers identified by Mean Shift, the Silhouette Score for K-Means optimization, and the final K-Means clustering.

![Lab 1 Clustering Results](https://github.com/user-attachments/assets/94d62c24-d316-4e99-87f1-7da296c25045)

#### 🧠 What I Learned {#lab1-what-i-learned}

This lab provided a hands-on understanding of **unsupervised learning** through two distinct clustering algorithms: **K-Means** and **Mean Shift**. I learned the fundamental difference between these approaches: K-Means is a centroid-based algorithm that requires the number of clusters ($k$) to be specified beforehand, while Mean Shift is a density-based, non-parametric algorithm that automatically discovers the number of clusters based on the data's density distribution.

Crucially, the exercise highlighted the importance of cluster evaluation. By implementing the **Silhouette Score**, I gained practical experience in determining the optimal number of clusters for the K-Means algorithm, ensuring the resulting clusters are well-separated and dense.

#### 💡 Use Cases {#lab1-use-cases}

| Industry/Area | Specific Application | Description |
| :--- | :--- | :--- |
| **Marketing** | Customer Segmentation | Grouping customers based on purchasing behavior, demographics, or website activity to tailor marketing campaigns. |
| **Finance** | Fraud Detection | Identifying unusual patterns or groups of transactions that deviate from the norm, flagging them as potential fraud. |
| **Image Processing** | Image Segmentation/Compression | Grouping pixels with similar colors or textures to simplify images or reduce the number of distinct colors for compression. |
| **Biology** | Gene Expression Analysis | Clustering genes with similar expression patterns to understand biological processes and identify co-regulated genes. |

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

#### 📈 Results {#lab2-results}

The genetic algorithm successfully optimized the objective function $f(x, y, z) = 1 / (1 + (x-2)^2 + (y+1)^2 + (z-1)^2)$, which has a theoretical maximum value of 1. The console output showed the evolution of the population over 100 generations, with the maximum fitness value steadily increasing. The **Best Individual** found was approximately $[2.0698, -0.8648, 0.9052]$, with a fitness of $0.9688$, which is very close to the theoretical maximum.

#### 🧠 What I Learned {#lab2-what-i-learned}

This lab served as an introduction to **evolutionary computation** and the practical implementation of a **Genetic Algorithm (GA)** using the `DEAP` framework. I learned how to model a complex optimization problem by defining the core components of a GA:
1.  **Individuals and Population**: Representing potential solutions and the collection of these solutions.
2.  **Fitness Function**: A crucial component that quantifies the quality of a solution, which the algorithm aims to maximize or minimize.
3.  **Genetic Operators**: Implementing selection (choosing the fittest individuals), crossover (combining two parents to create offspring), and mutation (introducing random changes) to drive the population toward an optimal solution.

The process demonstrated how GAs can effectively search a large, complex solution space to find near-optimal solutions for problems where traditional calculus-based methods are difficult or impossible to apply.

#### 💡 Use Cases {#lab2-use-cases}

| Industry/Area | Specific Application | Description |
| :--- | :--- | :--- |
| **Engineering** | Design Optimization | Optimizing the shape, material, or structure of components (e.g., airplane wings, car chassis) for maximum performance or minimal cost. |
| **Logistics** | Traveling Salesman Problem (TSP) / Routing | Finding the most efficient route for delivery trucks, service technicians, or network packets. |
| **Finance** | Portfolio Optimization | Selecting a mix of assets that maximizes expected return for a given level of risk. |
| **Artificial Intelligence** | Hyperparameter Tuning | Automatically searching for the best combination of hyperparameters for a machine learning model. |

**Key Concepts:**

*   Genetic Algorithms fundamentals
*   Fitness function definition
*   Selection, Crossover, and Mutation operators
*   `DEAP` framework usage

**Key Files:**

*   `main.py`: Python script containing the genetic algorithm implementation.

### Lab3: Neural Networks (MNIST Classification)

This lab demonstrates the implementation of a simple neural network using `Keras` for classifying handwritten digits from the MNIST dataset. It covers data loading, preprocessing (normalization, one-hot encoding), model creation, training, evaluation, and saving the trained model.

#### 📸 Screenshots {#lab3-screenshots}

The model was trained for 10 epochs, achieving a high level of accuracy on the test set. The final evaluation showed a test loss of **~0.0993** and a test accuracy of **~97.14%**. The visualization below shows an example of a test image, the model's prediction, and the actual label, confirming the model's ability to correctly classify the handwritten digit.

![Lab 3 Prediction Example](https://github.com/user-attachments/assets/7347926c-64e4-4b67-8619-cd0e80c1561d)

#### 🧠 What I Learned {#lab3-what-i-learned}

This lab provided foundational experience in **deep learning** by implementing a simple **Neural Network** for the classic MNIST handwritten digit classification task using **Keras**. Key takeaways include:
1.  **Data Preprocessing**: The necessity of normalizing pixel data (scaling to 0-1) and converting labels to a **one-hot encoded** format for categorical classification.
2.  **Model Architecture**: Building a sequential model with a dense input layer, a hidden layer with a **ReLU** activation function, and an output layer with a **Softmax** activation function for probability distribution over the 10 classes.
3.  **Training and Evaluation**: Understanding the role of the **Adam optimizer** and **Categorical Cross-entropy loss** in the training process, and evaluating the model's performance using accuracy on a separate test set.
4.  **Model Persistence**: The practical step of saving the trained model (`.keras` format) for later use without needing to retrain.

#### 💡 Use Cases {#lab3-use-cases}

| Industry/Area | Specific Application | Description |
| :--- | :--- | :--- |
| **Healthcare** | Medical Image Analysis | Classifying X-rays, MRIs, or CT scans to detect diseases like cancer or pneumonia. |
| **Security** | Facial Recognition | Identifying individuals from images or video streams for access control or surveillance. |
| **Retail** | Product Recommendation Systems | Predicting which products a customer is likely to purchase based on their past behavior and product features. |
| **Optical Character Recognition (OCR)** | Digitizing Documents | Converting handwritten or printed text in documents into machine-readable text. |

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
