module github.com/sparavec/devai-tools

// Consolidated to the current stable Go release across both modules
// (see gpu-arbiter/go.mod). Since platform-rebuild Phase 5 it requires the
// router's module for the shared bus contract and Slurm REST client
// (gpu-arbiter/pkg/bus, pkg/slurmrest; I2), from the working tree through
// the relative replace below: both modules are built with the host's Go
// from this repository. The router depends on nothing here.
go 1.27

require (
	github.com/nats-io/nats.go v1.54.0
	github.com/sparavec/gpu-arbiter v0.0.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/klauspost/compress v1.20.0 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/minio/highwayhash v1.0.3 // indirect
	github.com/nats-io/jwt/v2 v2.7.3 // indirect
	github.com/nats-io/nats-server/v2 v2.10.27 // indirect
	github.com/nats-io/nkeys v0.4.16 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	golang.org/x/crypto v0.57.0 // indirect
	golang.org/x/sys v0.48.0 // indirect
	golang.org/x/time v0.10.0 // indirect
)

replace github.com/sparavec/gpu-arbiter => ../gpu-arbiter
