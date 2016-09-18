This is a D port of nicodjimenez minimal lstm python example found at
http://github.com/nicodjimenez/lstm

You get the best performance when compiling with with ldc in release mode.
The according dub call would be:


```
    dub -b ldmd2
```

**NOTE** The required version to compile mir (a library we use for this example)
is **1.1.0-beta2** (at this time, anyway). Check
https://github.com/ldc-developers/ldc for more information


I found that the python example takes **0.18s** to run and with the proper
optimization the D example takes **0.11s**.
