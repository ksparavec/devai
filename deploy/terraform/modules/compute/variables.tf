# Common interface for compute across all clouds

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "image_uri" {
  description = "Container image URI"
  type        = string
}

variable "cpu" {
  description = "CPU units (vCPUs)"
  type        = number
  default     = 2
}

variable "memory" {
  description = "Memory in MB"
  type        = number
  default     = 4096
}

variable "port" {
  description = "Container port to expose"
  type        = number
  default     = 8888
}

variable "enable_gpu" {
  description = "Enable GPU support"
  type        = bool
  default     = false
}

variable "gpu_type" {
  description = "GPU type (cloud-specific)"
  type        = string
  default     = ""
}

variable "replicas" {
  description = "Number of replicas"
  type        = number
  default     = 1
}

variable "environment_variables" {
  description = "Environment variables for the container"
  type        = map(string)
  default     = {}
}

variable "subnet_ids" {
  description = "Subnet IDs for the compute resources"
  type        = list(string)
  default     = []
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
