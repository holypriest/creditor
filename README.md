# Creditor

A HTTP server that makes real-time default risk predictions based on multiple versions of a (dummy) Logistic Regression model.

## Overview

Creditor uses a widely-spread approach to serve data via HTTP with good latency results: A _Flask_ app, _Gunicorn_ as the
application server and _Nginx_ as the reverse-proxy server.

[Flask](http://flask.pocoo.org/) is a well-known minimalistic and lightweight microframework.
Together with [Gunicorn](http://gunicorn.org/), models can be served in a fast and reliable way. On top of that,
[Nginx](https://www.nginx.com/) is a battle-tested way to receive the requests and distribute them among the
Gunicorn workers efficiently.

The logic employed is stateless. As an initial setup, the model is trained and serialized to the disk. When a request
is received at `/predict`, the server loads it into memory so it can predict default risk for the input data. After
predicted data is sent back to the client, the model's in-memory representation is left to the garbage collector.
For the next request, it will get loaded from the disk again.

The serialization-into-disk strategy works well with logistic regression models, considering that they are small
(~ 800 bytes for the given LR model), no matter the size of the training data (it's just the coefficients of an equation,
after all). For models that store the training set data (such as kNN) or huge neural network models, maybe a caching strategy
should be employed (using Redis for caching is a common approach). For reloading these tiny, frequently accessed files,
the operational system caching does its job.

When more data are available, it is possible to update the model by making a request to `/update/<model_id>`,
where `<model_id>` states for any valid string (the ones that does not contains slashes or whitespaces).
A `.parquet` file for the updated dataset must be sent as form data, with the proper headers. The server
is configured to receive files sized up to 250 MB, but it is possible to change this behavior editing `nginx.conf`.
This HTTP file transfer approach was meant to be simple. In an insecure environment, one should consider using data
encryption or some other way to upload files to the server. Requiring a token authentication could also be added as an
extra layer of security.

The new created model will, then, receive requests at `predict/<model_id>`. The endpoint `/predict` (without a `<model_id>`)
will always serve the model given by `CREDITOR_MAIN_MODEL` environment variable, that can be set to point to any existing
version of the model you want.

A simple payload content validation was also implemented. If the input data schema differs the schema previously defined,
an error code will be thrown. This validation accounts for lacking fields, extra fields and wrong types.

Changes to the given model code (`model.py`) were intentionally kept to the minimum, only the necessary to fit
it to the approach. This way, Pandas warnings on views and copies of dataframes were ignored.

A list of versions with model id, creation timestamp and area under the ROC curve can be obtained by making a GET request
at `/versions`. The service running status can be monitored by a proper tool making a GET request to `/healthcheck`.

## Main files

- `creditor.py`: Main flask module, containing the views. This module also contains healthcheck, error handling and auxiliary
functions that are used only for creditor.

- `creditor_tests.py`: Testing. All endpoints were tested at least once, for positive and negative responses. Healthcheck,
error handling, auxiliary functions and validation schema were also tested.

- `model.py`: The given logistic regression model (with minor changes).

- `models/.version`: Version control file. Holds information for existent models, such as id, creation timestamp and
ROC AUC score.

- `nginx.conf`: A simple configuration file for Nginx.

- `setup.py`: A configuration script that runs on first server startup. It creates the main model and adds it to
the version control file if there are no other versions.

- `supervisord.conf`: Configuration file for _Supervisor_. It keeps Gunicorn and Nginx servers online, restarting them
if needed.

- `wsgi.py`: Gunicorn application object. Imports creditor app to be served by the gunicorn workers.


## Environment variables

```
CREDITOR_MAIN_MODEL=
CREDITOR_MAIN_DATAFILE=
```

For examples, please read the [Dockerfile](./Dockerfile).

## Running

Firstly, make sure you have Docker installed and running properly (for reference, I tested it with Docker for Mac
`17.12.0-ce-mac46` and Docker for Linux `17.12.0-ce`). Then, check if there is no service listening on port 80 or
if the port it not firewalled.

After that, you can build and run a ready-for-use container executing the line below (do not forget do `cd` to the
project folder, where the `Dockerfile` lies):

```sh
docker build -t creditor . && docker run -d -p 80:80 creditor
```

Now Creditor should accept requests at port 80. Ubuntu needs `sudo` privileges to execute `docker` command by default.

## Testing

```
pip install -r requirements.txt
python creditor_tests.py
```

Gitlab CI is configured to run the tests and get line coverage for `creditor.py` and `/schemas/payload_schema.py`.

## Performance

The server was deployed to a MacBook Pro (i7 2.2 GHz Quad-core, 16 GB RAM), listening at the local network.
Performance tests, using the `wrk` tool, show good latency results for 100 opened connections (30 seconds workload)
requesting at `/predict` (using the given example input as payload):

```
Running 30s test @ http://192.168.1.11
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    88.34ms    5.29ms 108.78ms   94.22%
    Req/Sec   281.52     24.18   353.00     72.82%
  Latency Distribution
     50%   88.49ms
     75%   90.07ms
     90%   91.90ms
     99%   96.46ms
  16454 requests in 30.06s, 3.81MB read
Requests/sec:    547.30
Transfer/sec:    129.86KB
```

Gunicorn showed best results when deployed with 9 workers (CPUs * 2 + 1), the `-w` parameter at:

```
gunicorn --bind 0.0.0.0:5000 wsgi:app -w 9 -t 90 -c /deploy/creditor/setup.py
```

This approach can be vertically and horizontally scaled up. The hardware could be improved (MacBooks are not the best
server material) and more machines could be added, with load balancing. Gunicorn and Nginx configurations could
also be tweaked for best performance in the available hardware.
