# Split weight helper

- split weights from a checkpoint file of a model to a checkpoint file of each layer
  ```python mydrop.py <your_file_name>```

- convert the checkpoint files of each layer to csv(read the raw data out)
    > First create a file 'layer.names.txt' with all 
        names and shape of the layers in the same direcotry as the testing.py file.
    !!! Format should be the same as the example text file in this repo
    Hint: All layer names and shape will be printed to the console when running mydrop
    ```python testing.py

- test the result
    In model.py file the result is tested, you can refer to the file while testing 
    your own splitted weights