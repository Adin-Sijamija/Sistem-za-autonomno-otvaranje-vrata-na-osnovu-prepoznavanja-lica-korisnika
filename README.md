System for Autonomous Door Opening Based on User Facial Recognition
[AI, deep learning, machine learning, facial recognition, Arduino]

[English]
Final thesis after the first cycle of studies at the Faculty of Information Technologies (FIT) in Mostar.
The project encompasses the fields of artificial intelligence and computer vision with the goal of recognizing the faces of users who make up the system’s user database.

The Python programming language was used.

The system consists of two main parts:

A computer application

Arduino components installed on a model door

Within the computer application, the following processes are carried out:

Adding user images to the database

Extracting user faces from the saved images in the database — During this process, we also align the face horizontally (ensuring the eyes are at the same level) and discard inadequate images (e.g., those without a face or with multiple faces).

Encoding user faces — Converting user faces into 128-dimensional numerical data vectors.

Training a support vector machine (SVM) — Using the previously obtained encodings to train a new AI model that can compare new encodings with those stored in the user database and estimate the confidence level that the new encoding belongs to one of the registered users.

Recognizing users in video footage — The AI is used to detect faces in a video stream. It then draws a bounding box around the face, encodes it, and sends it to the recognition AI. All recognition results are stored in a deque list that keeps results from the last 50 frames.

If we are confident that the face belongs to a registered user, we trigger the door opening by sending a signal to the Arduino.

If the person is unknown, we begin recording a video clip for later review.

More detailed information about the system can be found in the accompanying Word documentation.
