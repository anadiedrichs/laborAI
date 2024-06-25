# laborAI

## Initial setup 

1. Clone this repository

```
cd laborAI
python -m venv .venv
```

2. Create a Python virtual environment

```
cd laborAI
python -m venv .venv
```

3. Install the requirements

```
$ pip install -r requirements.txt
```
   
4. In your terminal, activate your environment with one of the following commands, depending on your operating system.

```
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate

```
## Ollama

Install Ollama

```
curl -fsSL https://ollama.com/install.sh | sh
```

Run ollama server 

```
ollama serve &
```

Download model

```
ollama pull llama3-chaqa
```

## Before you run: pre-requirements


* Ollama must be running. Remember to open a terminal and start ollama

```
ollama serve &
```


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
