# Fed-Server
This folder contains only the server and it's computation configuration required to run FL.

#### How to Run?
- docker build -t IMAGE-NAME .
- docker run IMAGE-NAME ARGS

#### Args supported
Below are the arguments that are supported in the `server.py` file
- ip  - IP address it should run
- port - Port to run
- num_rounds - Number of Clients that is required
- fit_clients - Number of Fit Clients that is required