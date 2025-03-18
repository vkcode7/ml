# ml
### ML projects

#### How to create a virtual env in python?
To use venv in your project, in your terminal, create a new project folder, cd to the project folder in your terminal, and run the following command:
```bash
python<version> -m venv virtual-environment-name
  
mkdir projectA
  
cd projectA
  
python3.8 -m venv env
 ```

When you check the new projectA folder, you will notice that a new folder called env has been created. env is the name of our virtual environment, but it can be named anything you want.
<br>
Activate it using:
```bash
source env/bin/activate OR conda activate myvenv
  
deactivate:
deactivate
```
important commands:
```bash
  pip freeze > requirements.txt
  ~ pip install -r requirements.txt
```
To use this virtual env in Spyder: Preferences -> Python Interpreter<br>
https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment
  
Cloning a GIT reopo: 
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
```
James Briggs: https://github.com/jamescalam/transformers/tree/main
<br>
Learn next-generation NLP with transformers for sentiment analysis, Q&A, similarity search, NER, and more

### LangChain + Retrieval Local LLMs for Retrieval QA - No OpenAI!!!
this video go through how to use LangChain without Open AI for retrieval QA - Flan-T5, FastChat-T5, StableVicuna, WizardLM
https://www.youtube.com/watch?v=9ISVjh8mdlA

### Adapting LLMs

This list highlights various techniques that you can use to tailor the LLM to meet your specific needs:
- Prompt engineering is the process of optimizing text prompts to guide the LLM to generate pertinent responses.
- In-context learning (ICL) allows the model to dynamically update its understanding during a conversation, resulting in more contextually relevant responses. Data is fed within the prompt.
- Retrieval-augmented generation (RAG) combines retrieval and generation models to surface new relevant data as part of a prompt.
- Fine-tuning entails customizing a pretrained LLM to enhance its performance for specific domains or tasks.
- Reinforcement learning from human feedback (RLHF) is the ongoing approach to fine-tuning in near real time by providing the model with feedback from human evaluators who guide and improve its responses.


