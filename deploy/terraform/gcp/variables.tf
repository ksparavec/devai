# Common variables (same interface across all clouds)

variable "project_name" {
  description = "Name of the project (for resource naming)"
  type        = string
  default     = "devai-lab"
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "region" {
  description = "GCP region"
  type        = string
}

# Resource configuration
variable "cpu" {
  description = "CPU cores"
  type        = number
  default     = 2
}

variable "memory" {
  description = "Memory in MB"
  type        = number
  default     = 4096
}

variable "storage_size_gb" {
  description = "Storage size in GB"
  type        = number
  default     = 50
}

variable "replicas" {
  description = "Number of replicas (min instances for Cloud Run)"
  type        = number
  default     = 1
}

# Features
variable "enable_gpu" {
  description = "Enable GPU support (requires GKE, not Cloud Run)"
  type        = bool
  default     = false
}

variable "enable_https" {
  description = "Enable HTTPS"
  type        = bool
  default     = true
}

# Network
variable "allowed_cidrs" {
  description = "CIDR blocks allowed to access the service"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Container
variable "image_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}
