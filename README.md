# 🩺 Forward Neural Network for Diabetes Prediction

This project focuses on building a forward neural network for predicting diabetes using a given dataset. The tasks include understanding the data, setting up a model for training, fitting the model, improving the model, analyzing learned representations, and investigating neural collapse in deep learning.

## 📚 Table of Contents
- [Group 1: Construct a Forward Neural Network](#group-1-construct-a-forward-neural-network)
  - [Task 1.1: Understanding the Data](#task-11-understanding-the-data)
  - [Task 1.2: Setting Up a Model for Training](#task-12-setting-up-a-model-for-training)
  - [Task 1.3: Fitting the Model](#task-13-fitting-the-model)
- [Group 2: Improve the Model](#group-2-improve-the-model)
  - [Task 2.1: Check the Training Using TensorBoard](#task-21-check-the-training-using-tensorboard)
  - [Task 2.2: Apply Regularisation](#task-22-apply-regularisation)
- [Group 3: Analyze the Learned Representations](#group-3-analyze-the-learned-representations)
- [Group 4: Investigating Neural Collapse in Deep Learning](#group-4-investigating-neural-collapse-in-deep-learning)
  - [Task 4.1: Research and Understand the Concept of Neural Collapse](#task-41-research-and-understand-the-concept-of-neural-collapse)
  - [Task 4.2: Reproduce the NC Results](#task-42-reproduce-the-nc-results)
- [🚀 How to Run the Files](#how-to-run-the-files)
- [📬 Contact](#contact)
- [📜 License](#license)

## Group 1: Construct a Forward Neural Network

### Task 1.1: Understanding the Data
- Describe the target classes for the prediction task.
- Display 10 training examples from each target class. Identify any patterns.
- Describe the data types of each feature.
- Identify necessary preprocessing steps and their reasons.
- Prepare the data for learning a neural network, including creating training, validation, and test datasets.
- Specify the number of training and test examples used.

### Task 1.2: Setting Up a Model for Training
- Construct a deep feedforward neural network using only fully connected (dense) layers.
- Report configurations:
  - Output layer: number of nodes and activation function.
  - Hidden layers: number of layers, nodes in each layer, and activation functions.
  - Input layer: input size and whether reshaping is necessary.
- Justify model design decisions.
- Plot the model structure using `keras.utils.plot_model` or similar tools.

### Task 1.3: Fitting the Model
- Decide and report:
  - Loss function.
  - Metrics for model evaluation.
- Explain roles in model fitting.
- Decide the optimizer and report:
  - Training batch size.
  - Number of training epochs.
  - Learning rate and any additional configurations (momentum, learning rate schedule, etc.).
- Justify decisions.
- Fit the model and show how training loss and evaluation metrics change.
- Explain the criteria for stopping training.

## Group 2: Improve the Model

### Task 2.1: Check the Training Using TensorBoard
- Use TensorBoard to visualize the training process.
- Provide screenshots of TensorBoard output.
- Identify any signs of overfitting or underfitting and the epoch at which they occur.

### Task 2.2: Apply Regularisation
- Improve training by applying regularization techniques:
  - Dropout
  - Batch normalization
- Compare the effects of different regularization techniques.
- Optionally, explore other techniques such as learning rate scheduling.

## Group 3: Analyze the Learned Representations
- Visualize embeddings at different layers of the trained neural network using UMAP.
- Select a subset of training data containing both classes.
- Extract embeddings from each layer for the dataset.
- Apply UMAP to visualize embeddings in a 2D space, highlighting classes with distinct colors/markers.
- Analyze and discuss the evolution of embeddings across layers:
  - Clear separation between classes at any specific layer.
  - Changes in separation and clustering of classes across layers.
  - Notable changes in embedding distribution or structure.
  - Layers where embeddings become less discriminative or more entangled.
- Summarize findings and provide insights into the network's representation behavior.

## Group 4: Investigating Neural Collapse in Deep Learning

### Task 4.1: Research and Understand the Concept of Neural Collapse
- Understand the problem addressed by the neural collapse paper and its relation to the unit content.
- Explain how authors validate their proposed method or hypothesis.

### Task 4.2: Reproduce the NC Results
- Reproduce experiments from the neural collapse paper.
- Compare obtained results with those in the paper and identify any discrepancies.
- Discover connections between the paper and unit content.

## 🚀 How to Run the Files

1. Clone the repository:
    ```sh
    git clone https://github.com/syed-muqtasid-ali/diabetes-prediction.git
    ```

2. Navigate to the project directory:
    ```sh
    cd diabetes-prediction
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

5. Open the `Forward Neural Network.ipynb` file and execute the cells to follow along with the tasks.

6. Additionally, you can view the detailed project explanation in the `forward neural network.pdf` file.

## 📬 Contact
For any questions or inquiries, feel free to contact me via LinkedIn:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/syed-muqtasid-ali-91a0a623a/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:muqtasid5266@gmail.com)

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
Happy Learning! 😊
