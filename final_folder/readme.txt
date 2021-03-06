              _                                  _      _   _                               _           _   
             | |                                | |    | | (_)                             (_)         | |  
   __ _ _   _| |_ ___   ___ ___  _ __ ___  _ __ | | ___| |_ _  ___  _ __    _ __  _ __ ___  _  ___  ___| |_ 
  / _` | | | | __/ _ \ / __/ _ \| '_ ` _ \| '_ \| |/ _ \ __| |/ _ \| '_ \  | '_ \| '__/ _ \| |/ _ \/ __| __|
 | (_| | |_| | || (_) | (_| (_) | | | | | | |_) | |  __/ |_| | (_) | | | | | |_) | | | (_) | |  __/ (__| |_ 
  \__,_|\__,_|\__\___/ \___\___/|_| |_| |_| .__/|_|\___|\__|_|\___/|_| |_| | .__/|_|  \___/| |\___|\___|\__|
                                          | |                              | |            _/ |              
                                          |_|                              |_|           |__/               

Daniela Schacherer, Marvin Klaus, Sebastian Bek

requirements:

python packages:
-flask
-pytorch
-matplotlib
-itertools
-heapq
-random
-numpy
-easydict

'pred.pt' and 'text_gen.pt' (saved in epoch 6 and 12) are differing pretrained models for 2 purposes:
- char/word prediction
- text generation

The dataset is an altered (pre-processed) version of Friedrich Nietzsche's "Beyond Good and Evil" split into training and 
test dataset. The pre-processing routine removes all special characters from the initial text book.

'config.txt' is a simple config file with standard parameter variables configured for training the network from scratch,
which will be included on execution of the python script. In case you want to use a pretrained network, set the pretrained 
option to 'True'.

Final results can be observed by executing the python script 'autocompletion.py'.

To start the graphical user interface run the file ui/backend.py. The execution will start a server on localhost:5000. Please make sure that this port is free to use.
To use the graphical user interface go to this site and use the text input to send a request to the server and compute word suggestions.
With the gear symbol in the top right corner you can set the desired number of word suggestions.