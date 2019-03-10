# Split weight helper

- split weights from a checkpoint file of a model to a checkpoint file of each layer
  - ```python mydrop.py <your_file_name>```
  - After runinng this, you will find all checkpoint files in the output folder and a layer_names.txt

- convert the checkpoint files of each layer to csv(read the raw data out)
    - Hint: All layer names and shape will be printed to the console when running mydrop
    - ```python restoreCSV.py```

- test the result
    In model.py file the result is tested, you can refer to the file while testing 
    your own splitted weights