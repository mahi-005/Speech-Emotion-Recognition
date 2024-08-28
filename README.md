# Speech Emotion Recognition System - Sound Classification

## Project Overview

This project focuses on developing a Speech Emotion Recognition (SER) system that can classify emotions from speech audio. Using a Long Short-Term Memory (LSTM) network, the system classifies emotions into seven categories: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.

The dataset used in this project consists of 2800 audio files in WAV format, where two actresses of different ages (26 and 64 years) recorded the phrase "Say the word _" with each of the seven emotions. The goal is to train a neural network model to accurately classify the emotions based on the audio features extracted from these recordings.

## Dataset Information

The dataset contains 2800 audio files, divided between two female actors, each portraying the seven emotions. The data is organized into folders by actor and emotion, with 200 target words recorded in each emotion per actor.

### Emotion Categories

- Anger
- Disgust
- Fear
- Happiness
- Pleasant Surprise
- Sadness
- Neutral

### Dataset Structure

The dataset is organized into folders as follows:


/dataset
  /Actor_01
    /Anger
      - file1.wav
      - file2.wav
      ...
    /Disgust
    /Fear
    /Happiness
    /Pleasant_Surprise
    /Sadness
    /Neutral
  /Actor_02
    /Anger
    ...


### Download Links

- [Dataset (Toronto Emotional Speech Set - TESS)](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)

## Project Workflow

1. *Data Preprocessing*
   - Load the audio files using the librosa library.
   - Extract features such as Mel Frequency Cepstral Coefficients (MFCCs), Chroma, and Mel Spectrogram from each audio file.
   - Normalize the features for consistent input to the neural network.

2. *Model Development*
   - *Neural Network Architecture*: The model is built using a Long Short-Term Memory (LSTM) network due to its effectiveness in processing sequential data like audio.
   - *Libraries Used*:
     - pandas: Data handling and manipulation.
     - matplotlib: Visualization of data and model performance.
     - keras and tensorflow: Building and training the LSTM network.
     - librosa: Audio processing and feature extraction.

3. *Model Training*
   - Split the dataset into training and testing sets.
   - Train the LSTM network on the extracted features.
   - Monitor the model's accuracy and loss during training.

4. *Model Evaluation*
   - Test the model on unseen data.
   - Evaluate the model's performance with accuracy metrics.
   - Achieved accuracy: *67.00%*

5. *Visualization*
   - Use matplotlib to visualize the training process, including the loss and accuracy curves.
   - Display confusion matrix to understand the classification performance across different emotions.

## Results

The LSTM network achieved an accuracy of *67.00%* in classifying the emotions from speech. The model shows promise in recognizing speech emotions, though there is room for improvement in both feature engineering and model architecture.

## Conclusion

This project successfully demonstrates the application of deep learning techniques in speech emotion recognition. The LSTM network, despite its moderate accuracy, is capable of capturing the temporal dynamics of audio signals necessary for emotion classification. Future improvements could involve experimenting with different neural network architectures, enhancing the feature extraction process, or using larger and more diverse datasets.

## Getting Started

### Prerequisites

Ensure you have the following Python libraries installed:

bash
pip install pandas matplotlib keras tensorflow librosa


### Running the Project

1. Download the dataset from the provided link.
2. Extract the dataset into the project directory.
3. Run the script to train the model:

bash
python train_model.py


4. The model will be saved after training. You can test the model on new audio samples using the provided testing script.

## References

- [Toronto Emotional Speech Set (TESS) Dataset](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)



---

This README file serves as a comprehensive guide to understanding, running, and reproducing the results of the Speech Emotion Recognition system developed using an LSTM network.
