BOX LINK FOR MODELS: [Here](https://app.box.com/s/1djpq18pk2nfpwhiz3xdp6t5xhhz5e52)

## Note for TA Grader

By default, running `test2.py` will use the MNIST dataset. If you have your own validation data to test this model against, please specify the `-ta` argument when running the module. For example:

```bash
py test2.py -ta Path/To/TA_test_data
```

The test data directory structure should be similar to the data.zip provided by the professor in Canvas under hw_solutions:

```
TA_test_data/
   test/
       1.png
       2.png
       ...
   test_label.txt
```

## RBF Parameters

RBF parameters are initialized using a directory (from MNIST) called "digits_jpeg" structured as follows:

```
digits_jpeg/
    0/
        img001-000001.jpeg
        img001-000002.jpeg
        ...
    1/
        img002-000001.jpeg
        img002-000002.jpeg
        ...
    2/
    ...
```