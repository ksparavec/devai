module github.com/sparavec/devai-tools

// Consolidated to the current stable Go release across both modules
// (see gpu-arbiter/go.mod). The module has no dependencies outside the
// standard library since the MCP server (and its go-sdk) was removed on
// 2026-10-07 (platform-rebuild Phase 1, M10).
go 1.27
