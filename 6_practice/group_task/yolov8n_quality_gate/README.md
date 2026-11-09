# Test the Model

1. Extract the ZIP file.
2. Create `test/good` and `test/bad` inside the extracted folder. Put your test images in the matching folders.
3. Open a terminal in the extracted folder and run:

   ```bash
   python -m pip install ultralytics
   python predict.py --source test
   ```

4. Send back `predictions.csv` and a screenshot of the terminal results.
