import tensorflow as tf  # now import the tensorflow module
import numpy

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float16)

rank1_tensor = tf.Variable(["test"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

#print(rank2_tensor.shape)

tensor1 = tf.ones([1,2,3]) # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1]) # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 tells the tensor to calculate the size of the dimension in that place
                                        # this will reshape the tensor to [3,3]
#print(tensor1)
#print(tensor2)
#print(tf.rank(tensor1))

# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32)
#print(tf.rank(tensor))
#print(tensor.shape)

three = tensor[0,2]
print(three)
row1 = tensor[0]
print(row1)
column1 = tensor[:, 0]
print(column1)
row2and4 = tensor[1::2]
print(row2and4)
column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3)

