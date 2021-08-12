###############################################################################################################################

##  "Two-stream Encoder-Decoder Network for Localizing Image Forgeries" 
- by Aniruddha Mazumdar and Prabin K. Bora, IIT Guwahati (Under revision in Journal of Visual Communication and Image 									Representation)		 #####################################




##  Installation

The code is implemented and tested using the following libraries/dependencies:
1) python=3.6, 2) Keras=2.2.4, 3) Tensorflow_gpu=2.3.1, 4) numpy=1.17.2, 5) scikit-image=0.15.0, 6) scikit-learn=0.21.3, 7) pip=19.2.3, 8) matplotlib=3.1.1, 9) imageio=2.6.0, 10) scipy=1.1.0, 11) opencv-python=4.1.2.30

Additionally, if you want to run the demo jupyter notebook file, please install jupyter notebook.

1. To install these dependencies, first install anaconda by following the official download instructions avialable at "https://docs.anaconda.com/anaconda/install/linux/"
2. Create the anaconda environment and install the dependencie inside the environment:
	conda env create -f environment.yml


## Training the network 

# Pre-training on Synthetic Dataset prepared by Bappy et al [1].

 1. Modify the path to the synthetic training images and masks and the path to store the trained model weights in the training code and run the code:
	python encoder_decoder_train_2_decoder_Synthetic_NIST.py

 2. Change the path to the training images and masks and the path to the weights of the model trained on, and set the final model weight path to be stored and run the training code:
	python encoder_decoder_train_2_decoder_Synthetic_NIST.py


# Testing
1. To compute the predicted masks from the trained model, set the full path filename of the trained model weights, and run the following inference code:
	python encoder_decoder_train_2_decoder_Synthetic_NIST_Test.py --input_image <test_image_filename> --output_path <path_to_store_results> 

If it runs successfully, it should save to output files, i.e. predicted_binary_mask.png and predicted_prob_mask.npy, inside the output_path folder.





[1] J. H. Bappy, C. Simons, L. Nataraj, B. Manjunath, A. K. Roy-Chowdhury, Hybrid lstm and encoder–decoder architecture for detection of image forgeries, IEEE Transactions on Image Processing 28 (2019) 3286–3300.



