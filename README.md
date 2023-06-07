# ml
ML projects

How to create a virtual env in python?
To use venv in your project, in your terminal, create a new project folder, cd to the project folder in your terminal, and run the following command:

python<version> -m venv virtual-environment-name
  
 mkdir projectA
  
 cd projectA
  
 python3.8 -m venv env
  
When you check the new projectA folder, you will notice that a new folder called env has been created. env is the name of our virtual environment, but it can be named anything you want.
  
Activate it using:
source env/bin/activate
  
deactivate:
deactivate

important commands:
  pip freeze > requirements.txt
  ~ pip install -r requirements.txt

To use this virtual env in Spyder:
  
Preferences -> Python Interpreter

https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment
  
Cloning a GIT reopo: git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
