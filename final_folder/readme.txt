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

The dataset is an altered version of Friedrich Nietzsche's "Beyond Good and Evil" split into training and test (validation)
dataset.