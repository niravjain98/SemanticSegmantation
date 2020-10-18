import tensorflow as tf
import helper
from functions import *


#Tune these parameters

NUMBER_OF_CLASSES = 2
IMAGE_SHAPE = (160,576)
EPOCHS = 40
BATCH_SIZE = 16
DROPOUT = 0.75

#Specify the directory paths

data_dir = './dataset'
runs_dir = './runs'
training_dir = './dataset/data_road/training'
vgg_path = './dataset/vgg'

#PLACEHOLDERS TENSORS

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES]) #SHAPE = [None,160,576,2]
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

#FUNCTIONS



def main():
  
  # Download pretrained vgg model
  #helper.maybe_download_pretrained_vgg(data_dir)

  # A function to get batches
  get_batches_fn = helper.gen_batch_function(training_dir, IMAGE_SHAPE)
  
  with tf.Session() as session:
        
    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)

    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)

    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)
    
    # Initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    print("Model about to train, Nigga")

    # Train the neural network
    # #train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, 
    #          train_op, cross_entropy_loss, image_input,
    #          correct_label, keep_prob, learning_rate)

    # # Run the model with the test images and save each painted output image (roads painted green)
    # helper.save_inference_samples(runs_dir, data_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input)
    
    print("All done!")

#--------------------------
# MAIN
#--------------------------
if __name__ == '__main__':
    main()

#FUNCTIONS

