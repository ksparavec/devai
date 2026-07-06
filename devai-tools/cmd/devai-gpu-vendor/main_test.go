package main

import "testing"

func TestVendorValuesCompleteness(t *testing.T) {
	for vendor, values := range vendorValues {
		for _, key := range orderedKeys {
			if values[key] == "" {
				t.Errorf("vendorValues[%q] missing %s", vendor, key)
			}
		}
		if values["DEVAI_GPU_VENDOR"] != vendor {
			t.Errorf("vendorValues[%q][DEVAI_GPU_VENDOR] = %q, want %q", vendor, values["DEVAI_GPU_VENDOR"], vendor)
		}
	}
	if len(vendorValues) != 2 {
		t.Errorf("vendorValues has %d entries, want exactly nvidia and amd", len(vendorValues))
	}
}

func TestVendorDeviceStringsAreVendorSpecific(t *testing.T) {
	if vendorValues["nvidia"]["DEVAI_GPU_DEVICE"] != "nvidia.com/gpu=all" {
		t.Errorf("nvidia device = %q", vendorValues["nvidia"]["DEVAI_GPU_DEVICE"])
	}
	if vendorValues["amd"]["DEVAI_GPU_DEVICE"] != "amd.com/gpu=all" {
		t.Errorf("amd device = %q", vendorValues["amd"]["DEVAI_GPU_DEVICE"])
	}
}
