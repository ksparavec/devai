# Common outputs for storage
# Each cloud implementation provides these outputs

output "storage_id" {
  description = "ID of the storage resource"
  value       = ""
}

output "mount_target" {
  description = "Mount target for the storage (DNS name or IP)"
  value       = ""
}

output "volume_handle" {
  description = "Volume handle for Kubernetes CSI"
  value       = ""
}
