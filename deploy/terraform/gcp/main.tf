# GCP Infrastructure for DevAI Lab
# Uses Cloud Run for serverless container deployment

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  # GCP naming: lowercase, alphanumeric, hyphens
  safe_name   = lower(replace(local.name_prefix, "_", "-"))
  common_labels = {
    project     = var.project_name
    environment = var.environment
    managed-by  = "terraform"
  }
}

# =============================================================================
# Enable required APIs
# =============================================================================

resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

# =============================================================================
# Artifact Registry
# =============================================================================

resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = local.safe_name
  description   = "Docker repository for ${var.project_name}"
  format        = "DOCKER"

  labels = local.common_labels

  depends_on = [google_project_service.artifactregistry]
}

# =============================================================================
# Service Account
# =============================================================================

resource "google_service_account" "cloudrun" {
  account_id   = "${local.safe_name}-run"
  display_name = "Cloud Run Service Account for ${var.project_name}"
}

resource "google_project_iam_member" "cloudrun_artifactregistry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.cloudrun.email}"
}

# =============================================================================
# Cloud Run Service
# =============================================================================

resource "google_cloud_run_v2_service" "main" {
  name     = local.safe_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloudrun.email

    scaling {
      min_instance_count = var.replicas
      max_instance_count = var.replicas * 3
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.main.repository_id}/${var.project_name}:${var.image_tag}"

      ports {
        container_port = 8888
      }

      resources {
        limits = {
          cpu    = "${var.cpu}"
          memory = "${var.memory}Mi"
        }
      }

      env {
        name  = "JUPYTER_PORT"
        value = "8888"
      }
    }
  }

  labels = local.common_labels

  depends_on = [
    google_project_service.run,
    google_artifact_registry_repository.main
  ]
}

# =============================================================================
# IAM - Allow public access (or restrict based on allowed_cidrs)
# =============================================================================

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = contains(var.allowed_cidrs, "0.0.0.0/0") ? 1 : 0
  location = google_cloud_run_v2_service.main.location
  name     = google_cloud_run_v2_service.main.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# For restricted access, use IAP or Cloud Armor instead
# This is a simplified example for development
