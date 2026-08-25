# granger-test — Go 时间序列 Granger 因果检验与 VAR 建模 HTTP 服务

Time-series Granger causality testing service. Implements VAR model estimation,
lag order selection (AIC/BIC), F-test and chi-square causality statistics,
impulse response functions, cointegration tests, and spectral analysis.

## Build / Run / Test

```bash
go build -o granger-test .
./granger-test serve --addr :8080
./granger-test -x data/x.csv -y data/y.csv -lag 4
go test ./...
```

## Evaluation Image

Evaluation-specific files (do not overwrite project Dockerfile/README):

- `benzhi.Dockerfile`
- `build_benzhi_docker.sh`
- `BENZHI_README.md` (this file)

Build and verify in container:

```bash
chmod +x build_benzhi_docker.sh
./build_benzhi_docker.sh <image-name> linux/arm64
./build_benzhi_docker.sh <image-name> linux/amd64
docker run -it <image-name>:latest
```
