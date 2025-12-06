# GCP Terraform outputs

output "service_url" {
  description = "URL to access the Cloud Run service"
  value       = google_cloud_run_v2_service.main.uri
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.main.repository_id}"
}

output "artifact_registry_name" {
  description = "Artifact Registry repository name"
  value       = google_artifact_registry_repository.main.repository_id
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.cloudrun.email
}

output "cloud_run_service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.main.name
}

output "push_command" {
  description = "Command to push image to Artifact Registry"
  value       = <<-EOT
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker ${var.region}-docker.pkg.dev

    # Build and push
    docker build -t ${var.project_name} .
    docker tag ${var.project_name}:latest ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.main.repository_id}/${var.project_name}:latest
    docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.main.repository_id}/${var.project_name}:latest
  EOT
}
