# run_pipeline.py
from pipeline.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline(test_size=0.2)



