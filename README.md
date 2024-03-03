# Indian Sign Language (Dell: Hack2Hire)

## Introduction

This application is a sophisticated real-time hand gesture recognition system. It leverages the power of computer vision and machine learning to detect and classify hand gestures in real-time using the webcam feed.

## Features

- **Real-time hand gesture recognition**: The application can recognize hand gestures in real-time using the webcam feed.
- **Hand detection and classification**: The application uses the `cvzone` library for hand detection and classification.
- **Gesture classification**: The application can classify gestures into numbers 1-9 and the letters A-Z.
- **Web interface**: The application provides a user-friendly web interface for interaction.

## Installation

Clone the repository: `git clone https://github.com/amnullh/Indian_Sign_Language-DELL.git`

## Usage

To run the application, use the command: `python main.py`

Once the application is running, navigate to `localhost:5001` in your web browser.

## API Endpoints

- `/` : Home page
- `/module` : Module page
- `/test` : Test page
- `/learn` : Learn page
- `/predict` : Returns the predicted hand gesture as a JSON response
- `/pred_ans` : Returns the predicted answer as a JSON response
- `/download_report` : Downloads the report of the test
- `/video` : Returns the video feed with hand gesture recognition

