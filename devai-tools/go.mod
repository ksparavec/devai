module github.com/sparavec/devai-tools

// 1.25, not gpu-arbiter's 1.23: github.com/modelcontextprotocol/go-sdk
// (cmd/devai-mcp-modelstatus) requires go >= 1.25.0. Old code compiles
// fine under the higher directive; only the build container's Go tag
// needs to match (see Makefile's build-mcp-modelstatus /
// test-devai-tools targets: golang:1.25-bookworm, not 1.23).
go 1.25.0

require gopkg.in/yaml.v3 v3.0.1

require (
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/modelcontextprotocol/go-sdk v1.6.1
	github.com/segmentio/asm v1.1.3 // indirect
	github.com/segmentio/encoding v0.5.4 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/oauth2 v0.35.0 // indirect
	golang.org/x/sys v0.41.0 // indirect
)
