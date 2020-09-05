def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

  keep_prob_value = 0.5
  learning_rate_value = 0.001
  for epoch in range(epochs):
      # Create function to get batches
      total_loss = 0
      for X_batch, gt_batch in get_batches_fn(batch_size):

          loss, _ = sess.run([cross_entropy_loss, train_op],
          feed_dict={input_image: X_batch, correct_label: gt_batch,
          keep_prob: keep_prob_value, learning_rate:learning_rate_value})

          total_loss += loss;

      print("EPOCH {} ...".format(epoch + 1))
      print("Loss = {:.3f}".format(total_loss))
      print()