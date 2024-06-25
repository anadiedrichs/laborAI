# laborAI

### Initial setup 

1. Create a Python virtual environment

```
cd laborAI
python -m venv .venv
```

2. Install the requirements

```
$ pip install -r requirements.txt
```
   
3. In your terminal, activate your environment with one of the following commands, depending on your operating system.

   
```
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate

```
### Before you run: pre-requirements


* Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct to ask for access.

* Visit huggingface.co and get your token


### How to run it on your own machine

* Run the app

```
$ streamlit run app.py
```

### Stop the application

* To stop the Streamlit server, press Ctrl+C in the terminal.

* When you're done using this environment, return to your normal shell by typing:

```
$ deactivate
```
