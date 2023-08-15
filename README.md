# forward-neural-network
You are going to build a neural network for the image classification task. You will train the model on the diabetes prediction dataset.
Group 1 Construct a forward neural network
(weight ~60%)

With this group of tasks, you are going to build a neural network for the image classification task. You will train the model on the diabetes prediction dataset.

# Task 1.1 Understanding the data
(weight ~20%)

Describe the target classes for the prediction task. Display 10 training examples from each target class. Do you see any patterns?
Describe the data types of each feature. What preprocessing steps are required? Why?
Prepare the data for learning a neural network, including creating training, validation, and test datasets. How many training examples and how many test examples are you using?
# Task 1.2 Setting up a model for training
(weight ~ 20%)

Construct a deep feedforward neural network. In other words, you can use only fully connected (dense) layers. You need to decide and report the following configurations:

Output layer:
How many output nodes?
Which activation function?
Hidden layers:
How many hidden layers?
How many nodes in each layer?
Which activation function for each layer?
Input layer
What is the input size?
Do you need to reshape the input? Why?
Justify your model design decisions.

Plot the model structure using keras.utils.plot_model or similar tools.

# Task 1.3 Fitting the model
(weight ~ 20%)

Decide and report the following settings:

The loss function
The metrics for model evaluation (which may be different from the loss function)
Explain their roles in model fitting.

Decide the optimiser that you will use. Also report the following settings:

The training batch size
The number of training epochs
The learning rate. If you used momentum or a learning rate schedule, please report the configuration as well.
Justify your decisions.

Now fit the model. Show how the training loss and the evaluation metric change. How did you decide when to stop training?

# Group 2 Improve the model
(weight ~ 10%)

# Task 2.1 Check the training using TensorBoard
Use TensorBoard to visualise the training process. Show screenshots of your TensorBoard output.

Do you see overfitting or underfitting? Why? If you see overfitting, at which epoch did it happen?

# Task 2.2 Apply regularisation
Improve the training process by applying regularisation. Below are some options:

Dropout
Batch normalisation
Compare the effect of different regularisation techniques on model training. You may also try other techniques for improving training such as learning rate scheduling (see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule).

# Group 3 Analyse the learned representations
(weight ~ 10%)

In this task, you will explore the visualization of embeddings at different layers of your trained neural network and analyse how they evolve using Uniform Manifold Approximation and Projection (UMAP). Below are detailed steps you can follow.

Select a subset of your training data containing both classes.
Extract the embeddings from each layer of the neural network model for the dataset.
Apply UMAP to visualise the embeddings from each layer in a 2-dimensional space, highlighting different classes with distinct colours or markers. Include appropriate labels and legends in your plots.
Analyse and discuss the evolution of the embeddings across layers. Answer the following questions in your analysis:
Do the embeddings show clear separation between classes at any specific layer?
How do the separation and clustering of classes change as you move across layers?
Are there any notable changes in the distribution or structure of the embeddings?
Are there any layers where the embeddings become less discriminative or more entangled?
Summarize your findings and provide insights into the behaviour of the neural network's representations at different layers. Discuss the implications of the observed changes in the embeddings for the network's ability to capture class-specific information and make predictions.
# Group 4 Investigating Neural Collapse in Deep Learning
(weight ~20%)

In this research task, you will explore the phenomenon of "neural collapse" in deep learning models.

# Task 4.1 Research and understand the concept of neural collapse in deep learning.
What problem does the paper address? How is it related to what you have learnt in the unit so far?
How do the authors validate their proposed method or hypothesis?
# Task 4.2 Reproduce the NC results
Reproduce experiments described in the neural collapse paper. Compare the results you obtained with the ones in the paper. Do you identify any discrepancies?

What connections do you discover between the paper and what you have learnt in the unit.
