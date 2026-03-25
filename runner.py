import sys
try:
    with open("runner_output.txt", "w") as out:
        sys.stdout = out
        sys.stderr = out
        
        from src.train import run_pipeline
        run_pipeline()
        print("PIPELINE SUCCESS")
except Exception as e:
    import traceback
    with open("runner_output.txt", "a") as err:
        traceback.print_exc(file=err)
