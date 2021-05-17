# ML-Image-Classifier
A convolutional neural networks (CNN)-based machine learning algorithm which incoporates TensorFlow and Keras to classify images
Trained with the CIFAR-100 dataset.

# Limitations
While CIFAR-10 is easy to work with, training on CIFAR-100 certainly poses some limitations as it is not easy to train a really deep learning model and is not feasible to get results as high as 90% such as in MNIST. However, this has really been an interesting challenge for me to see how far this dataset could go. To save training time, I have used a batch size of 250 with 10 epochs and 0.2 validation split, using ReLu as activation function. Due to this nature, after a series of trail, this model has only been able to give an overall accuracy of 37.95%.

![image](https://user-images.githubusercontent.com/77548862/118425852-a28b8b00-b6fc-11eb-8ade-1a94b269bc18.png)
![image](https://user-images.githubusercontent.com/77548862/118425857-a5867b80-b6fc-11eb-805d-caf88fffbb36.png)

# Examples
Expected: motorcycle

![image](https://user-images.githubusercontent.com/77548862/118425904-c058f000-b6fc-11eb-8ca2-2750d43d3e25.png)
![image](https://user-images.githubusercontent.com/77548862/118425924-cc44b200-b6fc-11eb-8343-ef3c041e0bc1.png)

Expected: tractor

![image](https://user-images.githubusercontent.com/77548862/118426256-7b818900-b6fd-11eb-995e-197331f3b908.png)
![image](https://user-images.githubusercontent.com/77548862/118426274-82a89700-b6fd-11eb-8ca4-007651759fc1.png)

Expected: lion

![image](https://user-images.githubusercontent.com/77548862/118426462-d31ff480-b6fd-11eb-8b94-ad6ee33de923.png)
![image](https://user-images.githubusercontent.com/77548862/118426474-dadf9900-b6fd-11eb-9618-b325d5c4b832.png)

Well.... it (kinda) works!

# Remedy
An alternate apporach would be the model used by Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter which uses ELU as activation function as re-implemented in this article https://medium.com/@birdortyedi_23820/deep-learning-lab-episode-5-cifar-100-a557e19219ba. While this model could give a top-1 accuracy of 75.72%, it would reuqire harsh requirements such as 165000 epochs. For the specs of my pc (2070) it is still expected to run at least 20 days non-stop. Anyway, I have learnt quite a lot in this project, and that's what important ^^.

# Credits
Inspired by Youtube Channel Computer Science https://www.youtube.com/watch?v=iGWbqhdjf2s
