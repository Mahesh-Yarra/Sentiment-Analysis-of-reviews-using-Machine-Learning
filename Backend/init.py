import subprocess

def run_model():
    # Run model.py using the Python interpreter
    subprocess.run(['python', 'model.py'])

def run_app():
    # Run app.py using the Python interpreter
    subprocess.run(['python', 'app.py'])

if __name__ == "__main__":
    # Execute the functions sequentially
    run_model()  # First, run model.py
    run_app()    # Then, run app.py
