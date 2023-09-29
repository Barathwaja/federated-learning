#### docker run -d --name prometheus -v $(pwd):/etc/config -p 9090:9090 prom/prometheus --config.file=/etc/config/prometheus.yaml

#### docker run -d -p 3000:3000 --name=grafana grafana/grafana-enterprise