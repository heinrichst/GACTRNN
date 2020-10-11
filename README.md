# GACTRNN  


Stefan Heinrich et al. 2020  
Gated and Adaptive Continuous Time Recurrent Neural Networks 
heinrich.stefan@ircn.jp  


Reference:  
@inproceedings{Heinrich2020GACTRNN,  
	author       = {Heinrich, Stefan and Alpay, Tayfun and Nagai, Yukie},  
	title        = {Learning Timescales in Gated and Adaptive Continuous Time Recurrent Neural Networks},  
	booktitle    = {Proceedings of the 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC2020)},
	address      = {Toronto, CA},
	pages        = {2662--2667},  
	year         = {2020}  
}  


###Usage:  
Use the various CTRNN versions just like the other RNN models in **tensorflow** **keras** version 1.15. For documentation on variable connotation see xctrnn_cell.py. As a special characteristic the CTRNN variants are implemented in a modularised fashion already, which means instead of just a dense layer a CTRNN can consist of different partitions (called moduls) that are connected with different connectivity schemes. If a CTRNN is defined with an arbitrary number of modules and connected 'dense', then it will function as a normal fully connected layer.  


####Example:  
...  
import tensorflow as tf
from tensorflow import keras  
import models.keras_extend.xctrnn as keras_layers_xctrnn
...  
\#define your input
...
num_hidden_v = [30, 20, 10]  
tau_hidden_v = [1, 8, 64]  
net_connectivity = 'dense'  
initializer_w_tau = tf.glorot_normal_initializer()  
...  
ctrnn = keras_layers_xctrnn.GACTRNN(num_hidden_v, tau_vec=tau_hidden_v, connectivity=net_connectivity)
hidden = ctrnn(input)[0]
model = keras.Model(input, hidden)
...


###Alternative:
Use the prepared wrapper in the CTRNNs.py
