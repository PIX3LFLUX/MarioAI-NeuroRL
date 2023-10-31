import subprocess
import sys

if len(sys.argv) < 2:
	last_level = ""
	last_model = ""
else:	
	last_level = sys.argv[1]
	last_model = sys.argv[2]

# Define the levels and models to be tested
levels = ["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0", "SuperMarioBros-1-3-v0"]
models = ["model1", "model2", "model3", "model4"]

# Give Test_Models.py signal that called from Automatic_Mode.py
automatic_mode = "True"

# Choose right level and model for next run	
if last_level == "" and last_model == "":
	level = "SuperMarioBros-1-1-v0" 
	model = "../models/model1"
elif last_level == "SuperMarioBros-1-1-v0" and last_model == "model1":
	level = "SuperMarioBros-1-2-v0"
	model = "../models/1000it"
elif last_level == "SuperMarioBros-1-2-v0" and last_model == "1000it":
	level = "SuperMarioBros-1-3-v0"
	model = "../models/ea1000it"
elif last_level == "SuperMarioBros-1-3-v0" and last_model == "ea1000it":
	level = "SuperMarioBros-1-1-v0"
	model = "../models/model2"
elif last_level == "SuperMarioBros-1-1-v0" and last_model == "model2":
	level = "SuperMarioBros-1-2-v0"
	model = "../models/2000it"
elif last_level == "SuperMarioBros-1-2-v0" and last_model == "2000it":
	level = "SuperMarioBros-1-3-v0"
	model = "../models/ea2000it"
elif last_level == "SuperMarioBros-1-3-v0" and last_model == "ea2000it":
	level = "SuperMarioBros-1-1-v0"
	model = "../models/model3"
elif last_level == "SuperMarioBros-1-1-v0" and last_model == "model3":
	level = "SuperMarioBros-1-2-v0"
	model = "../models/8000it"
elif last_level == "SuperMarioBros-1-2-v0" and last_model == "8000it":
	level = "SuperMarioBros-1-3-v0"
	model = "../models/ea8000it"
elif last_level == "SuperMarioBros-1-3-v0" and last_model == "ea8000it":
	level = "SuperMarioBros-1-1-v0"
	model = "../models/model4"
elif last_level == "SuperMarioBros-1-1-v0" and last_model == "model4":
	level = "SuperMarioBros-1-2-v0"
	model = "../models/austrainiert"
elif last_level == "SuperMarioBros-1-2-v0" and last_model == "austrainiert":
	level = "SuperMarioBros-1-3-v0"
	model = "../models/eaaustrainiert"
elif last_level == "SuperMarioBros-1-3-v0" and last_model == "eaaustrainiert":
	level = "SuperMarioBros-1-1-v0"
	model = "../models/model1"
	
# Build the command to execute the Test_models.py script
command = ["python", "test_models.py", level, model, automatic_mode]
		
# Execute the command
subprocess.call(command)
		
sys.exit()
