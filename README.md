# federated-learning
This will consists of all the Federated Learning codes


#### How to RUN?
- cd server && docker build -t fl-server .
- cd client-1 && docker build -t fl-client-1 .
- cd client-2 && docker build -t fl-client-2 .

- docker network create --subnet 192.0.0.100/24 custom-net 

#### To run in Background
- docker run -d --network=custom-net --ip=192.0.0.101 --name=fl-server fl-server (Map your SERVER IP - as 192.0.0.101 and any port of your choice.)
- docker run -d --network=custom-net --ip=192.0.0.102 --name=fl-client-1 fl-client-1 (Here we map the SERVER IP - 192.0.0.101 and PORT you configured.)
- docker run -d --network=custom-net --ip=192.0.0.103 --name=fl-client-2 fl-client-2 (Here we map the SERVER IP - 192.0.0.101 and PORT you configured.)


#### To run in Interactive Mode
- docker run -it --network=custom-net --ip=192.0.0.101 fl-server /bin/bash
- docker run -it --network=custom-net --ip=192.0.0.102 fl-client-1 /bin/bash
- docker run -it --network=custom-net --ip=192.0.0.103 fl-client-2 /bin/bash

- docker run -it --network=custom-net --ip=192.0.0.101 fl-server-km /bin/bash
- docker run -it --network=custom-net --ip=192.0.0.102 fl-client-km-1 /bin/bash
- docker run -it --network=custom-net --ip=192.0.0.103 fl-client-km-2 /bin/bash