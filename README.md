handwriting-recognition
=======================

Python wrapper around a neural net (octave)

the application takes a webcam-photo every 0.3 seconds, processes and generates a dataset from the image, and then runs the dataset through a neural net.
using the output vector from the neural net, the applications makes a prediction on which (numeric) character can be seen through the webcam.


load_network.m also includes functions to generate random networks, and train networks to a given training set.

    note: if anyone wants to run this on their system, they will need to edit octave/octave_call.bat
