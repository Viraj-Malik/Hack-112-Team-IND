his is a repo for Hack 112 Spring 2024- Viraj Malik, Arnav Arora and Nisarg Banda

We have made an app that contains 2 games. The first one is the typical snake game and then second game is Flappy Bird.
The game uses facial recognition models to to determine which player is playing the games and sets the difficult of the games according to the user.
The facial recogniton model that we created was implemented using the k nearest neghibours algorithim using skylearn.
The current Model is trained on photos of Viraj Malik and Arnav Arora and each of these users have a different diffiuclty rating. (Note after rigorous testing we determined that the accuracy of our model is close to 75%.)
Player Detection:

This part of the application is designed to interact with the user’s webcam, allowing them to capture an image with a single click. Upon running, it presents a webcam window, prompting the user to capture a ‘selfie’.

This image is then saved to the local system, and is input into a facial recognition software that has been trained with various images of potential users to determine the identity of the current user.

The application automatically releases the webcam once the image has been captured.

The ‘Image Capture’ system uses openCV to start the webcam and capture the live video feed, with Python Imaging Library being used for real-time image processing.

Once the player is determined they can choose which game they want to play by inputing 1 or 2 into the terminal.

Snake Game:

This is just the normal version of the Snake Game like we did in class, just implemented in pygame as we couldnt get CMU_GRAPHICS to work locally.
Flappy Bird Additonal Features:

This version of Flappy Bird is quite special, implemented in pygame. We use input of hand movements and using object tracting determine the center of your hand/face and gives you control of your bird.

The algorithim we used for this tracking that is being used to move the bird was inspired from the CMU Perpetual Computing Lab and we integrated this into the flappy bird game to determine the y posistion of the bird.

For further implmentation the model can be trained onto the faces of 15-112 TAs and then when playing video games together some of the TAs don't have to feel bad if they are bad at the game as the difficulty can be set accoridng to user and face detection.

Citations:

PyGame Documentation
SkyLearn Documentation
Multiple Youtube Videos to learn how to train and use SkyLearn
Chat GPT to convert CMU_Graphics code to pygame for part of the snake game algorithim and give us outline to write Flappy Bird.
https://github.com/CMU-Perceptual-Computing-Lab/openpose For the Object Tracking idea which we incoperated into Flappy Bird
(https://www.youtube.com/watch?v=s6O5hlZ0QM4) for Image Capture.
