module github.com/sparavec/devai-tools

// Consolidated to the current stable Go release across both modules
// (see gpu-arbiter/go.mod) -- comfortably above the go >= 1.25.0 floor
// github.com/modelcontextprotocol/go-sdk (cmd/devai-mcp-modelstatus)
// itself requires.
go 1.26

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
