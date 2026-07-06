// Package modelcache reads devai's model catalog (deploy/models.yaml) and
// the probe/bench caches it is joined against, for the MCP model-status
// server's tools.
package modelcache

import (
	"os"

	"gopkg.in/yaml.v3"
)

// CatalogEntry is one deploy/models.yaml row. Only the fields the
// model-status tools actually consume are modeled; unknown YAML keys are
// ignored by yaml.v3's default unmarshal behavior.
type CatalogEntry struct {
	Name           string   `yaml:"name"`
	Family         string   `yaml:"family"`
	Backend        []string `yaml:"backend"`
	Repo           string   `yaml:"repo"`
	Source         string   `yaml:"source"`
	Sha            string   `yaml:"sha"`
	Size           string   `yaml:"size"`
	Purpose        string   `yaml:"purpose"`
	Conversational bool     `yaml:"conversational"`
}

type catalogFile struct {
	Models []CatalogEntry `yaml:"models"`
}

// LoadCatalog parses deploy/models.yaml.
func LoadCatalog(path string) ([]CatalogEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cf catalogFile
	if err := yaml.Unmarshal(data, &cf); err != nil {
		return nil, err
	}
	return cf.Models, nil
}

func hasBackend(backends []string, want string) bool {
	for _, b := range backends {
		if b == want {
			return true
		}
	}
	return false
}

func containsString(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
}
