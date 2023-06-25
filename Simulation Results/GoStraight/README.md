# Execution command

Enter the folder <strong>src</strong> and run the following command:
`python3 Test_walker.py -n <> -s <> -w <> -f <>`
For example, `python3 Test_walker.py -n 0.8 -s 1 -w trt -f 0.8`

# Parameters

- -n: The normalized ratio is a float in [0, 1].
- -s: There are two values: <strong>0</strong> means leg-based walking, and <strong>1</strong> means spine-based walking.
- -w: There are three walking gaits: `trt`, `walk` and `lat`
- -f: The stride frequency is a float in [0, 1.5].