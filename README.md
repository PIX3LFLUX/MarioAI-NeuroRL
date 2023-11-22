# MarioAI-NeuroRL
Neuroevolution and Reinforcement Learning for a Super Mario Bros Playing AI Agent

* this project is highly influenced from two major works
  1. [_Play Super Mario Bros with a Double Deep Q-Network_ by Andrew Grebenisan (Python Project)](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/)
  2. [_Evolution-Guided Policy Gradient in Reinforcement Learning_ by Shauharda Khadka and Kagan Tumer (Scientific Paper)](https://arxiv.org/abs/1805.07917)
* also note that chatGPT was used as a tool for programming 
* the whole project is written in python
* you can find all python files in the directory src
* all scripts (besides automatic_mode.py) can be started from the GUI, if you navigate to the project folder MarioAI-NeuroRL and then start the file GUI.py e.g. "python src/GUI.py"
* the file train_models.py can either be started directly e.g. via the batch command "python .../train_models.py" (environment is level 1) or via the GUI (select level for environment), same counts for test_models.py
* same for test_models.py if you start it directly (level 1 and 1000 iterations trained model1) human_player.py will also start in level 1
* the script human_player.py will start the environment of a level, then it waits for a human player to interact with it via Controller, if no Controller is connected (Bluetooth or USB) the script will crash
* in the directory models you can find pre-trained AI models for level 1,2 and 3, they will be used when the test_models.py script is started
* the automatic mode is a separate version of calling the test models script multiple times with different models automatically and infinetly
* in all scripts the Pygame-Window can be closed and the script cancelled when pressing the ESC-Key
* before running the Python scripts you have to setup the environment with the file requirements.txt (use "pip install -r requirements.txt" if you navigated to the MarioAI-NeuroRL directory  
* this project is tested with python 3.11.2 so make sure you got the same version
