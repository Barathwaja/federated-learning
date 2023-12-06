# Fed-Client
This folder contains only the client and it's computation configuration or algo required to run FL.

#### How to Run?
- docker build -t IMAGE-NAME .
- docker run IMAGE-NAME ARGS

#### Args supported
Below are the arguments that are supported in the `server.py` file
- ip  - IP address of the FL server
- port - Port of the FL server
- input_seq - The time-series window sequence length
- epochs - Number of iter.
- num_clusters - Number of clusters
- folder - Folder which contains the dataset that it should use for running algorithm under directory `data/geoaltitude/`

**NOTE: Change the ENTRYPOINT of python file to different algorithm and re-build new Docker Image**