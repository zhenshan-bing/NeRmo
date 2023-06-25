# Execution command

Enter the folder <strong>src</strong> and run the following command:
`python3 Test_turning.py -t <> -w <> -r <> -f <>`
For example, `python3 Test_turning.py -t mb -w trt -r 0.5 -f 0.8`

# Parameters

- -t: There are three turning modes. <strong>lb</strong> means leg-based turning, <strong>sb</strong> means spine-based turning, and <strong>mb</strong> means mix-based turning.
- -w: There are three walking gaits: `trt`, `walk` and `lat`
- -r: The turning rate will result in different turning radius for different gaits. This value is in [-1,1]
- -f: The stride frequency is a float in [0, 1.5].