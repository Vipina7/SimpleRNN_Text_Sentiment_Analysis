### Project Title: IMDB Movie Review Sentiment Analysis  

---

### Project Description:  

This project demonstrates the use of Simple Recurrent Neural Networks (RNNs) for sentiment analysis on text data. It uses the IMDB movie reviews dataset to train a model capable of classifying sentiments as positive or negative. The core components of the project include text preprocessing, encoding, model training, validation, and evaluation.

---

### Key Features:  
1. **Input Handling:**  
   - Text reviews are tokenized and encoded as sequences of word indices using Keras' IMDB dataset.  
   - Preprocessing includes padding sequences to a fixed length for uniform input shape.

2. **Text Decoding:**  
   - Encoded reviews are decoded back to text for better interpretability during testing and validation.

3. **Model Architecture:**  
   - A Simple RNN with an embedding layer is trained to classify sentiments.  
   - The model is configured with 128-dimensional embeddings, RNN layers, and a dense output layer with sigmoid activation for binary classification.

4. **Validation and Testing:**  
   - During training, 20% of the training data is used as a validation set to monitor performance and prevent overfitting.  
   - After training, the model is evaluated on an unseen test dataset to assess its generalization capability.

5. **Utility Functions:**  
   - Helper functions are provided to decode reviews, preprocess custom user input, and predict sentiment for new text.

---

### Repository Contents:  
- **`embedding.ipynb`**: Demonstrates text preprocessing and embedding representation.  
- **`simplernn.ipynb`**: Implements the RNN model for training and validation.  
- **`prediction.ipynb`**: Provides functions for decoding, preprocessing, and predicting sentiment on user-defined reviews.  
- **Saved Model (`simple_rnn_imdb.h5`)**: Pre-trained model for quick inference.  

This repository is a comprehensive starting point for understanding and experimenting with text sentiment analysis using Simple RNNs.